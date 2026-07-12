import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime
from util.get_job_ids import get_job_id
from util.search_terms import get_next_term, mark_run, compute_next_window
 
default_args = {
    "owner": "reyden",
    "retries": 1,
}
 
# Both are adjustable at runtime via `airflow variables set ...` (or the UI)
# without touching this file or redeploying. Tighten the schedule and/or
# shrink the increment while backfilling, then relax both once caught up.
DAG_SCHEDULE = Variable.get("ingest_and_chunk_schedule", default_var="@weekly")
BACKFILL_INCREMENT_DAYS = int(Variable.get("backfill_increment_days", default_var="90"))
 
 
def _get_next_term(**context):
    result = get_next_term()
    term = result["term"]
    window = compute_next_window(result["last_searched_through"], increment_days=BACKFILL_INCREMENT_DAYS)
 
    status = "steady-state" if window["caught_up"] else "backfilling"
    print(
        f"Next search term: {term} [{status}] "
        f"mindate={window['mindate']} maxdate={window['maxdate']} "
        f"(increment_days={BACKFILL_INCREMENT_DAYS})"
    )
 
    context["ti"].xcom_push(key="search_term", value=term)
    context["ti"].xcom_push(key="mindate", value=window["mindate"])
    context["ti"].xcom_push(key="maxdate", value=window["maxdate"])
    context["ti"].xcom_push(key="maxdate_iso", value=window["maxdate_iso"])
    return term
 
 
def _mark_run(**context):
    term = context["ti"].xcom_pull(task_ids="get_next_search_term", key="search_term")
    # Advance the watermark to the maxdate actually searched (not "today"),
    # so backfill runs advance by BACKFILL_INCREMENT_DAYS each time rather
    # than jumping straight to the present.
    searched_through = context["ti"].xcom_pull(task_ids="get_next_search_term", key="maxdate_iso")
    mark_run(term, searched_through)
 
 
with DAG(
    dag_id="ingest_and_chunk",
    default_args=default_args,
    schedule=DAG_SCHEDULE,
    start_date=datetime(2026, 6, 13),
    catchup=False,
    tags=["rag", "databricks"],
) as dag:
 
    get_next_search_term = PythonOperator(
        task_id="get_next_search_term",
        python_callable=_get_next_term,
    )
 
    ingest_pubmed = DatabricksRunNowOperator(
        task_id="ingest_pubmed",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("pubmed_ingestion_pipeline"),
        python_params=[
            "--query",
            "{{ ti.xcom_pull(task_ids='get_next_search_term', key='search_term') }}",
            "--max-results",
            "5000",
            "--mindate",
            "{{ ti.xcom_pull(task_ids='get_next_search_term', key='mindate') }}",
            "--maxdate",
            "{{ ti.xcom_pull(task_ids='get_next_search_term', key='maxdate') }}",
        ],
    )
 
    mark_search_term_run = PythonOperator(
        task_id="mark_search_term_run",
        python_callable=_mark_run,
    )
 
    chunk_abstracts = DatabricksRunNowOperator(
        task_id="chunk_abstracts",
        databricks_conn_id="databricks_default",
        job_id=get_job_id("abstract_chunking_pipeline"),
    )
 
    get_next_search_term >> ingest_pubmed >> mark_search_term_run >> chunk_abstracts