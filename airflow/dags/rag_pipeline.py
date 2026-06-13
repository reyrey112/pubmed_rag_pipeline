import sys
sys.path.append("/home/reyde/rag_pipeline")

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime
from airflow.providers.standard.operators.python import PythonOperator
from pipelines.pubmed_to_databricks import run_pipeline
from util.get_job_ids import get_job_id

default_args = {
    "owner": "reyden",
    "retries": 1,
}

with DAG(
    dag_id="rag_pipeline",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 6, 13),
    catchup=False,
    tags=["rag", "databricks"],
) as dag:

    ingest_pubmed = PythonOperator(
        task_id="ingest_pubmed",
        python_callable=run_pipeline,
        op_kwargs={"query": "Lentivirus", "max_results": 500},
    )

    chunk_abstracts = DatabricksRunNowOperator(
        task_id="chunk_abstracts",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("abstract_chunking_pipeline"),
    )

    embed_chunks = DatabricksRunNowOperator(
        task_id="embed_chunks",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("chunks_to_embeddings_pipeline"),
    )

    create_vector_index = DatabricksRunNowOperator(
        task_id="create_vector_index",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("vector_embedding_pipeline"),
    )

    ingest_pubmed >> chunk_abstracts >> embed_chunks >> create_vector_index
