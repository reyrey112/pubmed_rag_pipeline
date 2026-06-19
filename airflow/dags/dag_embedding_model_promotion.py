from airflow import DAG
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable
from datetime import datetime
import sys

sys.path.append("/home/reyde/rag_pipeline")
from util.get_job_ids import get_job_id
from util.production_configurations import update_config


def promote_best_model(**context):
    from databricks import sql
    import os

    conn = sql.connect(
        server_hostname=Variable.get("databricks_host"),
        http_path=Variable.get("databricks_http_path"),
        access_token=Variable.get("databricks_token"),
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model, model_path, embedding_dim, `hit_rate_at_5`
        FROM rag_pipeline.silver.embedding_eval_results
        ORDER BY evaluated_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print("No eval results found")
        return "no_action"

    # Best from the most recent evaluation batch
    best = max(rows, key=lambda r: r[3])
    best_model, best_path, best_dim, best_score = best

    current_model = Variable.get("embedding_model_name")
    current_score = float(Variable.get("embedding_model_hit_rate", default_var="0"))

    print(f"Current: {current_model} ({current_score:.3f})")
    print(f"Best candidate: {best_model} ({best_score:.3f})")

    if best_score > current_score:
        Variable.set("embedding_model_name", best_model)
        Variable.set("embedding_model_path", best_path)
        Variable.set("embedding_dimension", str(best_dim))
        Variable.set("embedding_model_hit_rate", str(best_score))
        print(f"Promoted {best_model}")

        update_config(
            {
                "embedding_model_name": best_model,
                "embedding_model_path": best_path,
                "embedding_dimension": int(best_dim),
            },
            updated_by="embedding_model_evaluation_and_promotion",
        )

        return "trigger_embed_and_vector"
    else:
        print("No improvement, keeping current model")
        return "no_action"


default_args = {"owner": "reyden", "retries": 1}

with DAG(
    dag_id="embedding_model_evaluation_and_promotion",
    default_args=default_args,
    schedule=None,  # trigger manually for now
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "evaluation"],
) as dag:

    run_evaluation = DatabricksRunNowOperator(
        task_id="run_embedding_evaluation",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("evaluate_embedding_models_pipeline"),
    )

    promote = BranchPythonOperator(
        task_id="promote_best_model",
        python_callable=promote_best_model,
    )

    trigger_embed_and_vector = TriggerDagRunOperator(
        task_id="trigger_embed_and_vector",
        trigger_dag_id="embed_and_vector",
    )

    no_action = EmptyOperator(task_id="no_action")

    run_evaluation >> promote >> [trigger_embed_and_vector, no_action]
