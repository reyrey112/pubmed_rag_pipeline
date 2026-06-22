# streamlit_app.py
from dotenv import load_dotenv

load_dotenv()
import sys
import os
import streamlit as st
import uuid

from rag_query_sparkless import (
    get_embed_model,
    get_model_and_tokenizer,
    get_vsc,
    rag_query,
)


def update_status(message):
    status_container.write(message)


current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
util_path = os.path.join(repo_root, "airflow", "dags", "util")
if util_path not in sys.path:
    sys.path.append(util_path)

from conversation_history import read_history, write_history

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id


# load models prior to start
@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        embed_model = get_embed_model()
        model, tokenizer = get_model_and_tokenizer()
        vsc = get_vsc()
    return embed_model, model, tokenizer, vsc


# Triggers on first page load — blocks until complete
load_models()

st.set_page_config(page_title="Pharma RAG", page_icon="🔬", layout="centered")

st.title("🔬 Pharma RAG")
st.caption("Ask questions grounded in biomedical research literature")

# state, conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# existing conversation rendering
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a research question..."):

    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Read conversation history from Delta
    history = read_history(session_id)

    # 3. Generate response with iterative retrieval + transparency
    with st.chat_message("assistant"):

        # Status container — shows live updates, collapses when done
        with st.status("Searching literature...", expanded=True) as status:

            # Status callback — called by iterative_retrieve() between iterations
            status_messages = []

            def update_status(message: str):
                status_messages.append(message)
                status.write(message)

            try:
                result = rag_query(
                    prompt,
                    history=history,
                    status_callback=update_status,
                )

                answer = result["answer"]
                sources = result["sources"]
                query_used = result["query_used"]
                chunk_ids = result["chunk_ids"]
                iterations = result["iterations"]
                queries_used = result["queries_used"]
                sufficient = result["sufficient"]
                final_reasoning = result["final_reasoning"]

                # Update status container label when done
                if sufficient:
                    status.update(
                        label=f"✅ Found answer after {iterations} search(es)",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(
                        label=f"⚠️ Answer generated with partial context ({iterations} searches)",
                        state="error",
                        expanded=False,
                    )

                # 4. Render answer
                st.markdown(answer)

                # 5. Render sources
                if sources:
                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                # 6. Render retrieval detail (expandable)
                with st.expander("🔎 Retrieval detail"):
                    st.markdown(f"**Iterations:** {iterations}")
                    st.markdown(f"**Sufficient context reached:** {sufficient}")
                    st.markdown(f"**Final reasoning:** {final_reasoning}")
                    st.markdown("**Queries used:**")
                    for i, q in enumerate(queries_used, 1):
                        st.markdown(f"{i}. {q}")

                # 7. Debug sidebar
                if os.environ.get("DEBUG", "false").lower() == "true":
                    with st.sidebar:
                        with st.expander("🔍 Last query used"):
                            st.caption(query_used)
                        with st.expander("🔁 All queries"):
                            for q in queries_used:
                                st.caption(q)

                # 8. Build full response for UI state
                sources_text = "\n".join([f"- {s}" for s in sources])
                full_response = f"{answer}\n\n**Sources:**\n{sources_text}"

                # 9. Write completed turn to Delta
                write_history(
                    session_id=session_id,
                    user_message=prompt,
                    assistant_response=answer,
                    query_used=str(queries_used),  # store all queries
                    chunks_retrieved=chunk_ids,
                )

            except Exception as e:
                status.update(
                    label="❌ Error during retrieval",
                    state="error",
                    expanded=True,
                )
                full_response = f"Error: {str(e)}"
                st.error(full_response)

    # 10. Update UI state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
