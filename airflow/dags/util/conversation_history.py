import os
import sys
from databricks import sql

HISTORY_TABLE = "rag_pipeline.silver.conversation_history"
DEFAULT_N_TURNS = 10
MAX_HISTORY_TURNS = 5


def _get_connection():
    # Establishes connection using standard environment variables
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )


def _get_next_turn_number(cursor, session_id: str) -> int:
    """Get the next turn number for this session."""
    cursor.execute(
        f"""
        SELECT COALESCE(MAX(turn_number), 0) + 1
        FROM {HISTORY_TABLE}
        WHERE session_id = %(session_id)s
    """,
        {"session_id": session_id},
    )
    row = cursor.fetchone()
    return row[0] if row else 1


def create_history_table() -> None:
    """Creates the conversation table if it is missing or empty.

    Returns:
        None
    """

    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                session_id          STRING,
                turn_number         INT,
                role                STRING,
                content             STRING,
                query_used          STRING,
                chunks_retrieved    STRING,
                created_at          TIMESTAMP
            )
        """)

    except Exception as e:
        print(f"An error occurred during table setup: {e}")
        raise e

    finally:
        cursor.close()
        conn.close()


def read_history(session_id: str, n_turns: int = DEFAULT_N_TURNS) -> list[dict]:
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT role, content
            FROM (
                SELECT role, content, created_at, turn_number
                FROM {HISTORY_TABLE}
                WHERE session_id = %(session_id)s
                ORDER BY turn_number DESC, created_at DESC
                LIMIT {n_turns}
            )
            ORDER BY turn_number ASC, created_at ASC
        """,
            {"session_id": session_id},
        )

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return [{"role": row[0], "content": row[1]} for row in rows]

    except Exception as e:
        # If table doesn't exist yet or read fails, return empty history so the app continues working without memory rather than crashing
        print(f"History read failed (non-fatal): {e}")
        return []


def enrich_query(new_message: str, history: list[dict]) -> str:
    """
    Combines conversation history with the new user message
    into an enriched query string for vector retrieval.

    Only includes prior USER messages (not assistant responses)
    since assistant text adds noise to embedding-based retrieval.

    Returns the raw new_message if no history exists.
    """
    if not history:
        return new_message

    # Extract most recent user turns from history
    user_turns = [msg["content"] for msg in history if msg["role"] == "user"]

    # Take last MAX_HISTORY_TURNS user messages
    recent_turns = user_turns[-MAX_HISTORY_TURNS:]

    if not recent_turns:
        return new_message

    # prior context first, new message last
    context_prefix = " ".join(recent_turns)
    enriched = f"{context_prefix} {new_message}".strip()

    print(f"Enriched query: {enriched}")
    return enriched


def query_from_history(
    new_message: str, session_id: str, n_turns: int = DEFAULT_N_TURNS
) -> str:
    """
    Convenience function that reads history and enriches the query in one call.
    This is what streamlit_app.py will call directly.
    """
    history = read_history(session_id, n_turns)
    return enrich_query(new_message, history)


def write_history(updates: dict):
    pass


if __name__ == "__main__":
    create_history_table()
