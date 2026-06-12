from databricks import sql
import os
import pandas as pd
from dotenv import load_dotenv


def write_to_delta_table(
    df: pd.DataFrame,
    meta_table: str,
    abstract_table: str,
):
    """Insert rows into a Databricks Delta table via SQL connector."""
    conn = sql.connect(
        server_hostname=f"{os.environ.get("DATABRICKS_HOST")}",
        http_path=f"{os.environ.get("DATABRICKS_HTTP_PATH")}",
        access_token=f"{os.environ.get("DATABRICKS_TOKEN")}",
    )

    cursor = conn.cursor()

    # make into function
    cursor.execute(
        f"CREATE SCHEMA IF NOT EXISTS {os.environ.get("DATABRICKS_CATALOG")}.bronze"
    )
    print(f"Schema {os.environ.get("DATABRICKS_CATALOG")}.bronze ready")

    # Create metadata table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {meta_table} (
            pmid       STRING,
            title      STRING,
            authors    STRING,
            journal    STRING,
            year       STRING,
            mesh_terms STRING,
            doi        STRING
        ) USING DELTA
    """)

    # Create Abstract Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {abstract_table} (
            pmid       STRING,
            abstract   STRING
        ) USING DELTA   
    """)

    # Batch insert
    print("Executing upload")
    meta_df = df.drop(["abstract"], axis=1)
    rows = [tuple(row) for row in meta_df.itertuples(index=False)]
    cursor.executemany(f"INSERT INTO {meta_table} VALUES (?,?,?,?,?,?,?)", rows)

    abstract_df = df.drop(
        ["title", "authors", "journal", "year", "mesh_terms", "doi"], axis=1
    )
    abstract_rows = [tuple(row) for row in abstract_df.itertuples(index=False)]
    cursor.executemany(f"INSERT INTO {abstract_table} VALUES (?,?)", abstract_rows)

    cursor.close()
    conn.close()
    print(f"Inserted {len(rows)} rows into {meta_table} and {abstract_table}")
