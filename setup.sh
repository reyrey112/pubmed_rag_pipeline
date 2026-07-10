#!/bin/bash
set -e

export AIRFLOW_HOME=~/rag_pipeline/airflow
echo 'export AIRFLOW_HOME=~/rag_pipeline/airflow' >> ~/.bashrc
source ~/.bashrc


if [ -f ~/rag_pipeline/.env ]; then
    # Automatically export all variables defined from this point forward
    set -o allexport
    source ~/rag_pipeline/.env
    # Turn off automatic exporting
    set +o allexport
else
    echo ".env file not found"
    exit 1
fi

airflow db migrate


databricks secrets create-scope rag_pipeline 2>/dev/null || echo "Scope already exists, skipping"
databricks secrets put-secret rag_pipeline GEMINI_API_KEY --string-value "$GEMINI_API_KEY"
databricks secrets put-secret rag_pipeline EMAIL --string-value "$EMAIL"

airflow connections delete databricks_default 2>/dev/null || true
airflow connections add 'databricks_default' \
    --conn-type 'databricks' \
    --conn-host "$DATABRICKS_HOST" \
    --conn-password "$DATABRICKS_TOKEN"


airflow variables set databricks_host "$DATABRICKS_HOST"
airflow variables set databricks_http_path "$DATABRICKS_HTTP_PATH"
airflow variables set databricks_token "$DATABRICKS_TOKEN"

echo "checking for production table"
EXIT_CODE=0
uv run "$HOME/rag_pipeline/airflow/dags/util/production_configurations.py" || EXIT_CODE=$?

# Only set Airflow variables if the python script exited with 0 (table did not exist/was empty)
if [ $EXIT_CODE -eq 0 ]; then
    echo "Production table did not exist or was empty. Setting default Airflow variables"
    airflow variables set embedding_model_name "all-MiniLM-L6-v2"
    airflow variables set embedding_model_path "/Volumes/rag_pipeline/silver/models/all-MiniLM-L6-v2"
    airflow variables set embedding_dimension "384"
    airflow variables set embedding_model_hit_rate "0"
    airflow variables set generation_model_name "google/flan-t5-base"
    airflow variables set generation_model_score "0"
    echo "Airflow variables successfully initialized."

elif [ $EXIT_CODE -eq 3 ]; then
    echo "Production table already exists with data. Skipping Airflow variable initialization."
else
    echo "Error: The production configuration check failed (Exit code: $EXIT_CODE)."
    exit $EXIT_CODE
fi

echo "checking for History table"
uv run "$HOME/rag_pipeline/airflow/dags/util/conversation_history.py" 

echo "checking for interview state table"
uv run "$HOME/rag_pipeline/airflow/dags/util/interview_state.py" 

echo "Setup complete. Starting Airflow."
airflow standalone