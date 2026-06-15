# generation_model_evaluation_dag.py

from airflow import DAG
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.models import Variable
from datetime import datetime
import sys

sys.path.append("/home/reyde/rag_pipeline")
from util.get_job_ids import get_job_id


def promote_best_generation_model(**context):
    from databricks import sql
    import os

    conn = sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model, composite_score
        FROM rag_pipeline.silver.generation_eval_results
        ORDER BY evaluated_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print("No eval results found")
        return "no_action"

    best_model, best_score = max(rows, key=lambda r: r[1])

    current_model = Variable.get("generation_model_name")
    current_score = float(Variable.get("generation_model_score", default_var="0"))

    print(f"Current: {current_model} ({current_score:.2f})")
    print(f"Best candidate: {best_model} ({best_score:.2f})")

    if best_score > current_score:
        Variable.set("generation_model_name", best_model)
        Variable.set("generation_model_score", str(best_score))
        print(f"Promoted {best_model}")
        return "promoted"
    else:
        print("No improvement, keeping current model")
        return "no_action"


default_args = {"owner": "reyden", "retries": 1}

with DAG(
    dag_id="generation_model_evaluation_and_promotion",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "evaluation", "generation"],
) as dag:

    run_evaluation = DatabricksRunNowOperator(
        task_id="run_generation_model_evaluation",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("evaluate_generation_models_pipeline"),
        python_params=["--sample_size", "20"],
    )

    promote = BranchPythonOperator(
        task_id="promote_best_generation_model",
        python_callable=promote_best_generation_model,
    )

    promoted = EmptyOperator(task_id="promoted")
    no_action = EmptyOperator(task_id="no_action")

    run_evaluation >> promote >> [promoted, no_action]