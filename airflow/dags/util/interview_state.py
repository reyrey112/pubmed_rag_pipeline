from dataclasses import dataclass, field
from typing import Optional
from gemini_call import gemini_call
import os
from dotenv import load_dotenv
from databricks import sql
import json

load_dotenv()
GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "howdy",
    "greetings",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "got it",
    "sounds good",
    "great",
    "perfect",
    "yes",
    "no",
    "sure",
    "alright",
    "cool",
}


INTERVIEW_CATEGORIES = [
    "system",
    "stage",
    "techniques",
    "constraints",
    "prior_work",
]

CATEGORY_LABELS = {
    "system": "Target system/molecule/organism",
    "stage": "Development stage",
    "techniques": "Available techniques and equipment",
    "constraints": "Constraints and requirements",
    "prior_work": "Prior work and known results",
}

CATEGORY_PROMPTS = {
    "system": (
        "Ask what specific molecule, organism, biological system, or material "
        "they are working with. Be specific to their research question."
    ),
    "stage": (
        "Ask what development stage they are at — e.g. early discovery, "
        "process optimization, scale-up, formulation development, "
        "clinical manufacturing, etc."
    ),
    "techniques": (
        "Ask what laboratory techniques, equipment, or analytical methods "
        "they have available or are comfortable using."
    ),
    "constraints": (
        "Ask about any key constraints — budget, timeline, regulatory "
        "requirements (GMP, GLP), material availability, or other limitations."
    ),
    "prior_work": (
        "Ask what they already know or have tried — prior experiments, "
        "known results, literature they have already reviewed, "
        "or approaches that have failed."
    ),
}

SKIP_COMMANDS = {"skip", "just search", "search now", "proceed", "continue"}

INTERVIEW_STATE_TABLE = "rag_pipeline.silver.interview_states"

_gemini_client = None


def _get_gemini_client():

    global _gemini_client
    if _gemini_client is None:
        from google import genai

        try:
            api_key = dbutils.secrets.get(scope="rag_pipeline", key="GEMINI_API_KEY")

        except NameError as e:
            api_key = os.environ.get("GEMINI_API_KEY")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_connection():
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )


def create_interview_state_table() -> None:
    """Creates the interview state table if it is missing or empty.

    Returns:
        None
    """
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_pipeline.silver.interview_states (
            session_id  STRING,
            state_json  STRING,
            created_at  TIMESTAMP,
            updated_at  TIMESTAMP
        );
    """)

    except Exception as e:
        print(f"An error occurred during table setup: {e}")
        raise e

    finally:
        cursor.close()
        conn.close()


def save_interview_state(session_id: str, state: "InterviewState"):
    """
    Persists the current interview state to Delta.
    Uses MERGE to upsert — one row per session_id.
    Called after each state change.
    """

    try:
        conn = _get_connection()
        cursor = conn.cursor()

        # Check if row exists for this session
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM {INTERVIEW_STATE_TABLE}
            WHERE session_id = %(session_id)s
        """,
            {"session_id": session_id},
        )

        exists = cursor.fetchone()[0] > 0

        state_json = json.dumps(state.to_dict())

        if exists:
            cursor.execute(
                f"""
                UPDATE {INTERVIEW_STATE_TABLE}
                SET state_json = %(state_json)s,
                    updated_at = current_timestamp()
                WHERE session_id = %(session_id)s
            """,
                {
                    "session_id": session_id,
                    "state_json": state_json,
                },
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO {INTERVIEW_STATE_TABLE}
                (session_id, state_json, created_at, updated_at)
                VALUES (%(session_id)s, %(state_json)s,
                        current_timestamp(), current_timestamp())
            """,
                {
                    "session_id": session_id,
                    "state_json": state_json,
                },
            )

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Interview state save failed (non-fatal): {e}")


def load_interview_state(session_id: str) -> "InterviewState | None":
    """
    Loads interview state from Delta for a given session ID.
    Returns None if no state exists for this session.
    Called on app startup to restore in-progress interviews.
    """
    from databricks import sql
    import json

    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT state_json FROM {INTERVIEW_STATE_TABLE}
            WHERE session_id = %(session_id)s
        """,
            {"session_id": session_id},
        )

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row is None:
            return None

        data = json.loads(row[0])
        return InterviewState.from_dict(data)

    except Exception as e:
        print(f"Interview state load failed (non-fatal): {e}")
        return None


def generate_interview_question(
    state: "InterviewState",
    category: str,
) -> str:
    """
    Generates a contextually appropriate clarifying question
    for the given category, informed by what has already been answered.

    Parameters:
        state     — current InterviewState with original question and answers so far
        category  — which category to ask about next

    Returns:
        A single natural language question string
    """
    # Build context from answers so far
    answers_so_far = ""
    if state.answers:
        lines = []
        for cat, answer in state.answers.items():
            label = CATEGORY_LABELS[cat]
            lines.append(f"  {label}: {answer}")
        answers_so_far = "\n".join(lines)
    else:
        answers_so_far = "None yet — this is the first clarifying question."

    category_instruction = CATEGORY_PROMPTS[category]

    prompt = f"""You are a scientific research assistant helping design an experiment.
A researcher has asked: "{state.original_question}"

Information already gathered:
{answers_so_far}

Your task: {category_instruction}

Rules:
- Ask ONE question only
- Keep it concise — one sentence
- Make it specific to their research question and what they have already told you
- Sound like a knowledgeable collaborator, not a form
- Do not number the question or add any preamble
- Respond with ONLY the question text

Question:"""

    client = _get_gemini_client()

    try:

        response = gemini_call(client, prompt)

        question = response.text.strip()

        print(f"question: {question}")

        # handle if it returns multiple questions
        if "?" in question:
            question = question[: question.index("?") + 1]

        print(f"pruned question: {question}")

        return question

    except Exception as e:
        # Fall back to a generic question for this category
        print(f"Question generation failed for {category}: {e}")
        fallback = {
            "system": "What specific molecule, organism, or system are you working with?",
            "stage": "What development stage are you currently at?",
            "techniques": "What techniques and equipment do you have available?",
            "constraints": "Are there any key constraints such as budget, timeline, or regulatory requirements?",
            "prior_work": "What have you already tried or what do you already know about this problem?",
        }
        return fallback[category]


def is_skip_command(user_message: str) -> bool:
    """
    Returns True if the user wants to skip remaining interview questions
    and proceed directly to retrieval with whatever context exists.
    """
    return user_message.strip().lower() in SKIP_COMMANDS


def should_start_interview(
    user_message: str,
    state: "InterviewState",
    history: list[dict],
) -> bool:
    """
    Determines whether a new user message should trigger
    the start of a new interview.

    Starts an interview when:
    - No interview is currently active
    - The message looks like a new research question (not a follow-up)
    - The message is long enough to be a real question (not a one-word reply)

    Parameters:
        user_message — the new message from the user
        state        — current InterviewState
        history      — conversation history so far

    Returns:
        True if a new interview should start
    """
    # Already in an active interview — don't start another
    if state.active:
        return False

    # Too short to be a research question
    if len(user_message.strip()) < 20:
        return False

    # No history — first message is always a research question
    if not history:
        return True

    # Has history — ask Claude if this is a new research question
    # or a follow-up to the existing conversation
    return _is_new_research_question(user_message, history)


def _is_new_research_question(
    user_message: str,
    history: list[dict],
) -> bool:
    """
    Asks Claude whether the message is a new research question
    or a follow-up to the existing conversation.
    Returns True if it's a new research question.
    """
    recent_history = history[-6:]  # last 3 turns
    history_text = "\n".join(
        [f"{msg['role'].upper()}: {msg['content'][:200]}" for msg in recent_history]
    )

    prompt = f"""Recent conversation:
{history_text}

New message: "{user_message}"

Is this new message a NEW research question that requires designing an experiment or study,
or is it a FOLLOW-UP to the existing conversation (clarification, continuation, or related question)?

Respond with ONLY one word: NEW or FOLLOWUP"""

    client = _get_gemini_client()

    try:
        response = gemini_call(client, prompt)

        result = response.text.strip().upper()
        return result == "NEW"

    except Exception as e:
        print(f"New question detection failed: {e}. defaulting to FOLLOWUP")
        return False


def build_retrieval_query(state: "InterviewState") -> str:
    """
    Builds a concise, embedding-friendly query string from
    the interview answers for use in vector search retrieval.

    Combines the original question with key terms from each answer
    to create a targeted retrieval query.

    Parameters:
        state — completed or partial InterviewState

    Returns:
        A natural language query string for vector search
    """
    client = _get_gemini_client()

    answers_text = state.summary()

    prompt = f"""Based on this research context, write a single concise search query
(2-3 sentences maximum) that would retrieve the most relevant scientific literature.
Focus on the specific technical terms, methods, and constraints mentioned.
Do not include general phrases like "research on" or "study of".
Respond with ONLY the query text.

Research context:
{answers_text}"""

    try:
        response = gemini_call(client, prompt)

        result = response.text.strip()
        return result

    except Exception as e:
        print(f"Retrieval query build failed: {e} — falling back to summary")
        return answers_text


def build_generation_context(state: "InterviewState") -> str:
    """
    Builds a detailed structured context string for the experiment
    design report generator.

    Formats all interview answers into a clear, structured prompt
    section that Claude can use to produce a tailored report.

    Parameters:
        state — completed or partial InterviewState

    Returns:
        A formatted multi-line context string
    """
    lines = [
        "RESEARCH CONTEXT",
        "=" * 40,
        f"Research question: {state.original_question}",
        "",
    ]

    for category in INTERVIEW_CATEGORIES:
        if category in state.answers:
            label = CATEGORY_LABELS[category]
            answer = state.answers[category]
            lines.append(f"{label}:")
            lines.append(f"  {answer}")
            lines.append("")

    # Note if interview was partial (user skipped)
    unanswered = [cat for cat in INTERVIEW_CATEGORIES if cat not in state.answers]
    if unanswered:
        missing = ", ".join([CATEGORY_LABELS[c] for c in unanswered])
        lines.append(f"Note: The following context was not provided: {missing}")
        lines.append("Generate the best possible design with available information.")

    return "\n".join(lines)

def is_greeting_or_formality(message: str) -> bool:
    """
    Returns True if the message is a greeting, acknowledgment,
    or other social formality that shouldn't trigger an interview.
    """
    cleaned = message.strip().lower().rstrip("!.,?")

    # Exact match against known greetings
    if cleaned in GREETINGS:
        return True

    # Short messages (under 15 chars) that aren't questions
    if len(cleaned) < 15 and "?" not in cleaned:
        return True

    return False

@dataclass
class InterviewState:
    """
    Tracks the state of the clarifying question interview
    for a single research question session.
    """

    original_question: str = ""
    asked_categories: list[str] = field(default_factory=list)
    questions_asked: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    complete: bool = False
    active: bool = False
    report_generated: bool = False

    def next_category(self) -> Optional[str]:
        """
        Returns the next category to ask about,
        or None if all categories have been asked.
        """
        for category in INTERVIEW_CATEGORIES:
            if category not in self.asked_categories:
                return category
        return None

    def record_question(self, category: str, question_text: str):
        """Records that a question was asked for a category."""
        if category not in self.asked_categories:
            self.asked_categories.append(category)
        self.questions_asked[category] = question_text

    def record_answer(self, category: str, answer_text: str):
        """Records the user's answer for a category."""
        self.answers[category] = answer_text

    def is_complete(self) -> bool:
        """
        Returns True if all categories have been answered.
        Updates self.complete as a side effect.
        """
        all_answered = all(cat in self.answers for cat in INTERVIEW_CATEGORIES)
        self.complete = all_answered
        return all_answered

    def answered_count(self) -> int:
        """How many categories have been answered so far."""
        return len(self.answers)

    def total_questions(self) -> int:
        """Total number of categories to ask about."""
        return len(INTERVIEW_CATEGORIES)

    def progress_label(self) -> str:
        """Human readable progress string for UI display."""
        return f"Question {self.answered_count() + 1} of {self.total_questions()}"

    def to_dict(self) -> dict:
        """Serialize to dict for st.session_state storage."""
        return {
            "original_question": self.original_question,
            "asked_categories": self.asked_categories,
            "questions_asked": self.questions_asked,
            "answers": self.answers,
            "complete": self.complete,
            "active": self.active,
            "report_generated": self.report_generated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterviewState":
        """Deserialize from dict (restoring from st.session_state)."""
        state = cls()
        state.original_question = data.get("original_question", "")
        state.asked_categories = data.get("asked_categories", [])
        state.questions_asked = data.get("questions_asked", {})
        state.answers = data.get("answers", {})
        state.complete = data.get("complete", False)
        state.active = data.get("active", False)
        state.report_generated = data.get("report_generated", False)

        return state

    def summary(self) -> str:
        """
        Returns a readable summary of all answers for use in
        context building and Delta table storage.
        """
        lines = [f"Research question: {self.original_question}"]
        for category in INTERVIEW_CATEGORIES:
            if category in self.answers:
                label = CATEGORY_LABELS[category]
                answer = self.answers[category]
                lines.append(f"{label}: {answer}")
        return "\n".join(lines)


if __name__ == "__main__":
    create_interview_state_table()
