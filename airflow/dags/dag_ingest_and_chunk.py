import sys

sys.path.append("/home/reyde/rag_pipeline")

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime
from util.get_job_ids import get_job_id

default_args = {
    "owner": "reyden",
    "retries": 1,
}

with DAG(
    dag_id="ingest_and_chunk",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 6, 13),
    catchup=False,
    tags=["rag", "databricks"],
) as dag:

    ingest_pubmed = DatabricksRunNowOperator(
        task_id="ingest_pubmed",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("pubmed_ingestion_pipeline"),
        python_params=["--query", "Viral vectors", "--max-results", "500"],
    )

    chunk_abstracts = DatabricksRunNowOperator(
        task_id="chunk_abstracts",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("abstract_chunking_pipeline"),
    )

    ingest_pubmed >> chunk_abstracts
