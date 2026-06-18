from databricks import sql
import os

CONFIG_TABLE = "rag_pipeline.silver.production_config"


def get_connection():
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )


def create_production_table():
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT COUNT(*) AS table_count
            FROM information_schema.tables
            WHERE table_catalog = 'rag_pipeline'
              AND table_schema   = 'silver'
              AND table_name     = 'production_config'
        """)
        table_already_existed = cursor.fetchone()[0] > 0

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_pipeline.silver.production_config (
                config_version        INT,
                updated_at            TIMESTAMP,
                updated_by            STRING,
                generation_model_name STRING,
                embedding_model_name  STRING,
                embedding_model_path  STRING,
                embedding_dimension   INT
            )
        """)

        if not table_already_existed:
            cursor.execute("""
                INSERT INTO rag_pipeline.silver.production_config VALUES (
                    1,
                    current_timestamp(),
                    'initial_setup',
                    'google/flan-t5-base',
                    'all-MiniLM-L6-v2',
                    '/Volumes/rag_pipeline/silver/models/all-MiniLM-L6-v2',
                    384
                )
            """)
            print("Table created and seeded with initial config.")
        else:
            print("Table already exists — skipping seed insert.")

    finally:
        cursor.close()
        conn.close()


def get_latest_config() -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT config_version, generation_model_name, embedding_model_name,
               embedding_model_path, embedding_dimension
        FROM {CONFIG_TABLE}
        ORDER BY config_version DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    return {
        "config_version": row[0],
        "generation_model_name": row[1],
        "embedding_model_name": row[2],
        "embedding_model_path": row[3],
        "embedding_dimension": row[4],
    }


def update_config(updates: dict, updated_by: str) -> int:
    """
    Insert a new config version, carrying forward unchanged fields
    from the latest row and overwriting only the keys in `updates`.
    """
    current = get_latest_config()
    next_version = current["config_version"] + 1

    merged = {
        "generation_model_name": current["generation_model_name"],
        "embedding_model_name": current["embedding_model_name"],
        "embedding_model_path": current["embedding_model_path"],
        "embedding_dimension": current["embedding_dimension"],
    }
    merged.update(updates)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO {CONFIG_TABLE}
        (config_version, updated_at, updated_by, generation_model_name,
         embedding_model_name, embedding_model_path, embedding_dimension)
        VALUES (%(version)s, current_timestamp(), %(updated_by)s,
                %(gen_model)s, %(emb_model)s, %(emb_path)s, %(emb_dim)s)
    """,
        {
            "version": next_version,
            "updated_by": updated_by,
            "gen_model": merged["generation_model_name"],
            "emb_model": merged["embedding_model_name"],
            "emb_path": merged["embedding_model_path"],
            "emb_dim": merged["embedding_dimension"],
        },
    )
    cursor.close()
    conn.close()

    print(f"Config v{next_version} written by {updated_by}: {merged}")
    return next_version


def rollback_to(config_version: int):
    """Re-apply an older config as a new version (full history preserved)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT generation_model_name, embedding_model_name,
               embedding_model_path, embedding_dimension
        FROM {CONFIG_TABLE}
        WHERE config_version = %(v)s
    """,
        {"v": config_version},
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        raise ValueError(f"config_version {config_version} not found")

    return update_config(
        {
            "generation_model_name": row[0],
            "embedding_model_name": row[1],
            "embedding_model_path": row[2],
            "embedding_dimension": row[3],
        },
        updated_by=f"rollback_to_v{config_version}",
    )

if __name__ == "__main__":
    create_production_table()