import os
from databricks import sql


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


if __name__ == "__main__":
    create_interview_state_table()
