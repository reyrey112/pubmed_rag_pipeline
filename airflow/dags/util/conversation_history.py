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


if __name__ == "__main__":
    create_history_table()
