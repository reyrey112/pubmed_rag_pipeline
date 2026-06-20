# streamlit_app.py
from dotenv import load_dotenv
load_dotenv()
import sys
import os
import streamlit as st

from rag_query_sparkless import get_embed_model, get_model_and_tokenizer, get_vsc, rag_query

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

st.set_page_config(
    page_title="Pharma RAG",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Pharma RAG")
st.caption("Ask questions grounded in biomedical research literature")

# state, conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# existing conversation rendering
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a research question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show response
    with st.chat_message("assistant"):
        with st.spinner("Searching literature..."):
            try:
                result = rag_query(prompt)
                answer = result["answer"]
                sources = result["sources"]

                # Render answer
                st.markdown(answer)

                # Render sources as expandable section
                if sources:
                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                # Build full response string for history
                sources_text = "\n".join([f"- {s}" for s in sources])
                full_response = f"{answer}\n\n**Sources:**\n{sources_text}"

            except Exception as e:
                full_response = f"Error: {str(e)}"
                st.error(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })