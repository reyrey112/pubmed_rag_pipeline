import sys
sys.path.append("/home/reyde/rag_pipeline")

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime
from util.get_job_ids import get_job_id
from airflow.models import Variable


default_args = {
    "owner": "reyden",
    "retries": 1,
}

with DAG(
    dag_id="embed_and_vector",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 6, 13),
    catchup=False,
    tags=["rag", "databricks"],
) as dag:

    embed_chunks = DatabricksRunNowOperator(
        task_id="embed_chunks",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("chunks_to_embeddings_pipeline"),
        python_params=[
            "--model_name", Variable.get("embedding_model_name"),
            "--model_path", Variable.get("embedding_model_path"),
        ],
    )

    create_vector_index = DatabricksRunNowOperator(
        task_id="create_vector_index",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("vector_embedding_pipeline"),
        python_params=[
            "--embedding_dim", Variable.get("embedding_dimension"),
        ],
    )

    embed_chunks >> create_vector_index
