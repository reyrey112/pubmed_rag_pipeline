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
    experiment_design_query,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
util_path = os.path.join(repo_root, "airflow", "dags", "util")
if util_path not in sys.path:
    sys.path.append(util_path)

from gemini_call import gemini_call
from conversation_history import read_history, write_history
from interview_state import (
    InterviewState,
    generate_interview_question,
    should_start_interview,
    is_skip_command,
    save_interview_state,
    load_interview_state,
    build_retrieval_query,
    is_greeting_or_formality,
)

_gemini_client = None


def _get_gemini_client():
    from google import genai

    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def update_status(message):
    status_container.write(message)


st.set_page_config(page_title="Pharma RAG", page_icon="🔬", layout="centered")

st.title("🔬 Pharma RAG")
st.caption("Ask questions grounded in biomedical research literature")


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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

if "interview_state" not in st.session_state:
    # Try to restore in-progress interview from Delta
    restored = load_interview_state(session_id)

    if restored and restored.active and not restored.complete:
        # Resume in-progress interview
        st.session_state.interview_state = restored.to_dict()
        st.toast("Resuming your previous research session", icon="🔄")
    else:
        # Fresh interview state
        st.session_state.interview_state = InterviewState().to_dict()


def get_interview_state() -> InterviewState:
    """Helper to deserialize InterviewState from session_state."""
    return InterviewState.from_dict(st.session_state.interview_state)


def set_interview_state(state: InterviewState):
    """Helper to serialize InterviewState back to session_state and Delta."""
    st.session_state.interview_state = state.to_dict()
    save_interview_state(session_id, state)


def run_experiment_design(interview_state: InterviewState):
    """
    Extracted Path C logic — runs experiment design query and renders report.
    Called from Path B after last answer, and directly on rerun if complete.
    """
    with st.chat_message("assistant"):
        # Status container — retrieval progress only
        with st.status("Building your experiment design...", expanded=True) as status:
            status_messages = []

            def update_status(message: str):
                status_messages.append(message)
                status.write(message)

            try:
                result = experiment_design_query(
                    state=interview_state,
                    status_callback=update_status,
                )

                report = result["report"]
                iterations = result["iterations"]
                sufficient = result["sufficient"]
                chunk_ids = result["chunk_ids"]
                queries_used = result["queries_used"]

                if sufficient:
                    status.update(
                        label=f"✅ Report complete — {iterations} search(es)",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(
                        label=f"⚠️ Report generated with partial context",
                        state="error",
                        expanded=False,
                    )

            except Exception as e:
                status.update(
                    label="❌ Error generating report",
                    state="error",
                    expanded=True,
                )
                st.error(str(e))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Error generating report: {str(e)}",
                    }
                )
                return

        # ── Report renders OUTSIDE status block — always visible ──────────────
        st.markdown("## 📋 Experiment Design Report")
        st.markdown(f"*Based on: {interview_state.original_question}*")
        st.divider()

        sections = [
            ("🔬 Background", "background"),
            ("🧪 Recommended Approach", "approach"),
            ("⚙️ Key Parameters", "parameters"),
            ("✅ Controls & Success Criteria", "controls"),
            ("⚠️ Risks & Mitigation", "risks"),
            ("🔭 Knowledge Gaps", "gaps"),
            ("📚 References", "references"),
        ]

        for label, key in sections:
            content = report.get(key, "")
            if content:
                with st.expander(label, expanded=(key == "background")):
                    st.markdown(content)

        with st.expander("🔎 Retrieval detail"):
            st.markdown(f"**Iterations:** {iterations}")
            st.markdown(f"**Sufficient context:** {sufficient}")
            st.markdown("**Queries used:**")
            for i, q in enumerate(queries_used, 1):
                st.markdown(f"{i}. {q}")

    # Mark report generated — future messages go to Path E
    interview_state.report_generated = True
    set_interview_state(interview_state)

    # Build text summary for history
    full_response = "\n\n".join(
        [
            f"**{label}**\n{report.get(key, '')}"
            for label, key in sections
            if report.get(key)
        ]
    )

    write_history(
        session_id=session_id,
        user_message=interview_state.original_question,
        assistant_response=full_response[:2000],
        query_used=str(queries_used),
        chunks_retrieved=chunk_ids,
    )

    interview_state.report_generated = True
    set_interview_state(interview_state)

    set_interview_state(InterviewState())

    st.session_state.messages.append(
        {"role": "assistant", "content": "Experiment design report generated above."}
    )


# UI conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Interview progress indicator ──────────────────────────────────────────────

interview_state = get_interview_state()
if interview_state.active and not interview_state.complete:
    st.progress(
        interview_state.answered_count() / interview_state.total_questions(),
        text=f"Gathering research context — {interview_state.progress_label()}",
    )
    st.caption("💡 Type **skip** to proceed directly to search with current context")

# ── Debug sidebar ─────────────────────────────────────────────────────────────
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
if DEBUG:
    with st.sidebar:
        st.caption(f"Session: {session_id[:8]}...")
        state = get_interview_state()
        st.caption(f"Interview active: {state.active}")
        st.caption(f"Interview complete: {state.complete}")
        st.caption(f"Answers: {len(state.answers)}/{state.total_questions()}")


# existing conversation rendering
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new user input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a research question..."):

    interview_state = get_interview_state()
    history = read_history(session_id)

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Path F: Greeting or formality ─────────────────────────────────────────
    if is_greeting_or_formality(prompt):
        with st.chat_message("assistant"):
            client = _get_gemini_client()
            response = gemini_call(client, prompt)
            reply = response.text
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # ── Path A: Skip command ──────────────────────────────────────────────────
    elif is_skip_command(prompt) and interview_state.active:
        interview_state.complete = True
        set_interview_state(interview_state)

        with st.chat_message("assistant"):
            msg = "Understood — proceeding with the context gathered so far."
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        run_experiment_design(interview_state)

    # ── Path B: Interview in progress — collect answer ────────────────────────
    elif interview_state.active and not interview_state.complete:
        # Record the answer to the last asked question
        last_category = (
            interview_state.asked_categories[-1]
            if interview_state.asked_categories
            else None
        )

        if last_category:
            interview_state.record_answer(last_category, prompt)

        # Check if interview is now complete
        if interview_state.is_complete():
            interview_state.complete = True
            set_interview_state(interview_state)

            with st.chat_message("assistant"):
                msg = (
                    "Thank you — I have everything I need. "
                    "Let me search the literature and build your experiment design..."
                )
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            # DON'T rerun — fall through to Path C below by triggering it directly
            run_experiment_design(interview_state)

        else:
            # Ask next clarifying question
            set_interview_state(interview_state)
            next_category = interview_state.next_category()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    next_question = generate_interview_question(
                        interview_state, next_category
                    )
                    interview_state.record_question(next_category, next_question)
                    set_interview_state(interview_state)
                    st.markdown(next_question)

            st.session_state.messages.append(
                {"role": "assistant", "content": next_question}
            )

    # ── Path C: Interview complete — run experiment design ────────────────────
    elif (
        interview_state.active
        and interview_state.complete
        and not interview_state.report_generated
    ):
        run_experiment_design(interview_state)

    # ── Path D: Start new interview ───────────────────────────────────────────
    elif should_start_interview(prompt, interview_state, history):
        # Initialize new interview state
        # new_state = InterviewState()
        interview_state.original_question = prompt
        interview_state.active = True

        # Generate first question
        first_category = interview_state.next_category()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                first_question = generate_interview_question(interview_state, first_category)
                interview_state.record_question(first_category, first_question)
                set_interview_state(interview_state)
                st.markdown(
                    "Great research question! I have a few clarifying questions "
                    "to help me find the most relevant literature and design "
                    "a tailored experiment for you.\n\n" + first_question
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": first_question}
        )

    # ── Path E: Regular RAG (follow-up after report, or non-research question)
    else:
        with st.chat_message("assistant"):
            # Status block with retrival progress
            result = None
            with st.status("Searching literature...", expanded=True) as status:
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

                    if result["sufficient"]:
                        status.update(
                            label=f"✅ Found answer after {result['iterations']} search(es)",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(
                            label=f"⚠️ Answer with partial context ({result['iterations']} searches)",
                            state="error",
                            expanded=False,
                        )

                except Exception as e:
                    status.update(
                        label="❌ Error during retrieval",
                        state="error",
                        expanded=True,
                    )
                    st.error(str(e))

            # Answer always visible outside of status block
            if result:
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📄 Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- {source}")

                with st.expander("🔎 Retrieval detail"):
                    st.markdown(f"**Iterations:** {result['iterations']}")
                    st.markdown(f"**Sufficient:** {result['sufficient']}")
                    st.markdown("**Queries used:**")
                    for i, q in enumerate(result["queries_used"], 1):
                        st.markdown(f"{i}. {q}")

                sources_text = "\n".join([f"- {s}" for s in result["sources"]])
                full_response = f"{result['answer']}\n\n**Sources:**\n{sources_text}"

                write_history(
                    session_id=session_id,
                    user_message=prompt,
                    assistant_response=result["answer"],
                    query_used=str(result["queries_used"]),
                    chunks_retrieved=result["chunk_ids"],
                )

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
