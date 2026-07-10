# Architecture

This document reflects the current repository layout and the implementation that is present in this workspace.

## Overview

This project is a PubMed-backed RAG pipeline for biomedical literature retrieval and question answering. The current implementation is centered on Databricks notebooks and Databricks job definitions rather than a separate top-level `pipelines/` package. The main flow is:

1. Ingest PubMed data into Databricks tables.
2. Chunk abstracts into smaller units.
3. Embed chunks with a sentence-transformer model.
4. Create or sync a Databricks Vector Search index.
5. Retrieve relevant chunks and generate answers through a query layer.
6. Evaluate embedding and generation models, then promote the best performers through Airflow.

The repository uses a medallion-style layout with `bronze`, `silver`, and `gold` concepts, although the currently implemented tables are primarily in `bronze` and `silver`.

---

## Repository structure

```text
rag_pipeline/
├── airflow/
│   └── dags/
│       ├── dag_embed_and_vector.py
│       ├── dag_embedding_model_promotion.py
│       ├── dag_generation_model_promotion.py
│       ├── dag_ingest_and_chunk.py
│       └── util/
│           ├── conversation_history.py
│           ├── gemini_call.py
│           ├── get_job_ids.py
│           ├── interview_state.py
│           ├── iterative_retrieval.py
│           └── production_configurations.py
├── databricks_jobs/
│   ├── job_abstract_to_chunks.py
│   ├── job_chunks_to_embeddings.py
│   ├── job_embeddings_to_vector.py
│   ├── job_evaluate_embedding_models.py
│   ├── job_evaluate_generation_models.py
│   ├── job_generate_evaluation_set.py
│   └── job_pubmed_to_databricks.py
├── databricks_notebooks/
│   ├── abstracts_to_chunks.py
│   ├── chunks_to_embeddings.py
│   ├── embeddings_to_vector.py
│   ├── gradio_chat.py
│   ├── pubmed_to_databricks.py
│   ├── rag_query.py
│   ├── rag_query_sparkless.py
│   ├── streamlit_app.py
│   └── vector_index_test.py
├── dbt/
├── model_testing_notebooks/
│   ├── evaluate_embedding_models.py
│   ├── evaluate_generation_models.py
│   └── generate_evaluation_set.py
├── steps/
│   ├── csv_to_databricks_volume.py
│   ├── mysql_to_csv.py
│   ├── volume_to_delta_table.py
│   └── __init__.py
├── .env (gitignored)
├── Architecture.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.sh
└── uv.lock
```

Notes:
- The `dbt/` folder is present but currently does not contain model files in this workspace.
- The repository uses notebook-style Python scripts as the primary implementation layer, with job definitions under `databricks_jobs/`.

---

## Data flow

### 1. Ingestion
- The main ingestion logic is implemented in `databricks_notebooks/pubmed_to_databricks.py`.
- It uses BioPython Entrez to search and fetch PubMed records, then writes metadata and abstracts to Databricks Delta tables.
- The corresponding Databricks job definition is `databricks_jobs/job_pubmed_to_databricks.py`.

### 2. Chunking
- `databricks_notebooks/abstracts_to_chunks.py` reads the abstract table and applies chunking with `RecursiveCharacterTextSplitter`.
- Chunk IDs are produced in the format `${pmid}_chunk_${chunk_index}`.
- Output is written to `rag_pipeline.silver.chunks`.

### 3. Embedding
- `databricks_notebooks/chunks_to_embeddings.py` embeds each chunk with a sentence-transformer model.
- The notebook expects the model name and model path to be passed as CLI arguments.
- Output is written to `rag_pipeline.silver.embeddings` with change-data-feed enabled.

### 4. Vector Search
- `databricks_notebooks/embeddings_to_vector.py` creates or updates the Vector Search endpoint and sync index.
- The default endpoint name is `rag_pipeline_endpoint`.
- The default index name is `rag_pipeline.silver.chunk_index`.

### 5. Query and UI
- `databricks_notebooks/rag_query.py` performs retrieval and generation.
- `databricks_notebooks/rag_query_sparkless.py` provides a Spark-less variant used by the Streamlit app.
- `databricks_notebooks/streamlit_app.py` is the current interactive app entry point.
- `databricks_notebooks/gradio_chat.py` is a notebook-based Gradio demo.

### 6. Evaluation and promotion
- Evaluation scripts live in `model_testing_notebooks/`.
- Airflow DAGs in `airflow/dags/` trigger evaluation runs and update production configuration.

---

## Key modules

### Ingestion
- `databricks_notebooks/pubmed_to_databricks.py`
  - Searches PubMed via Entrez.
  - Parses article metadata and abstracts.
  - Writes to bronze Delta tables.

### Chunking and embedding
- `databricks_notebooks/abstracts_to_chunks.py`
- `databricks_notebooks/chunks_to_embeddings.py`

### Vector indexing
- `databricks_notebooks/embeddings_to_vector.py`
- `databricks_notebooks/vector_index_test.py`

### Query layer
- `databricks_notebooks/rag_query.py`
- `databricks_notebooks/rag_query_sparkless.py`

### User interfaces
- `databricks_notebooks/streamlit_app.py`
- `databricks_notebooks/gradio_chat.py`

### Evaluation
- `model_testing_notebooks/generate_evaluation_set.py`
- `model_testing_notebooks/evaluate_embedding_models.py`
- `model_testing_notebooks/evaluate_generation_models.py`

### Airflow helpers
- `airflow/dags/util/get_job_ids.py`
- `airflow/dags/util/production_configurations.py`
- `airflow/dags/util/conversation_history.py`
- `airflow/dags/util/interview_state.py`
- `airflow/dags/util/iterative_retrieval.py`
- `airflow/dags/util/gemini_call.py`

### Utility helpers in `steps/`
- `steps/csv_to_databricks_volume.py`
- `steps/mysql_to_csv.py`
- `steps/volume_to_delta_table.py`

These scripts are useful for data movement and volume-based workflows, but they are not the primary ingestion path for the PubMed RAG pipeline.

---

## Databricks jobs

The repository defines the following Databricks jobs in `databricks_jobs/`:

- `pubmed_ingestion_pipeline`
  - Created by `job_pubmed_to_databricks.py`
- `abstract_chunking_pipeline`
  - Created by `job_abstract_to_chunks.py`
- `chunks_to_embeddings_pipeline`
  - Created by `job_chunks_to_embeddings.py`
- `vector_embedding_pipeline`
  - Created by `job_embeddings_to_vector.py`
- `generate_evaluation_set_pipeline`
  - Created by `job_generate_evaluation_set.py`
- `evaluate_embedding_models_pipeline`
  - Created by `job_evaluate_embedding_models.py`
- `evaluate_generation_models_pipeline`
  - Created by `job_evaluate_generation_models.py`

These job names are used by the Airflow DAGs through the helper in `airflow/dags/util/get_job_ids.py`.

---

## Airflow DAGs

The current DAGs are:

- `dag_ingest_and_chunk.py`
  - Runs ingestion and then the chunking job.
- `dag_embed_and_vector.py`
  - Runs embedding and vector index creation.
- `dag_embedding_model_promotion.py`
  - Evaluates embedding models and promotes winners.
- `dag_generation_model_promotion.py`
  - Evaluates generation models and promotes winners.

The Airflow configuration is initialized from `setup.sh` and the production configuration table managed by `production_configurations.py`.

---

## Data model and target tables

The current implementation is expected to work with the following Databricks objects:

- `rag_pipeline.bronze.pubmed_meta`
- `rag_pipeline.bronze.abstracts`
- `rag_pipeline.silver.chunks`
- `rag_pipeline.silver.embeddings`
- `rag_pipeline.silver.eval_questions`
- `rag_pipeline.silver.embedding_eval_results`
- `rag_pipeline.silver.generation_eval_results`
- `rag_pipeline.silver.production_config`

The production config table is the authoritative source for model selection in the query layer.

---

## Configuration and secrets

### Environment variables
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_CATALOG`
- `GEMINI_API_KEY`
- `EMAIL`

These values are expected to be present in the local `.env` file or in the runtime environment.

### Databricks secrets
- Scope: `rag_pipeline`
- Example secrets: `GEMINI_API_KEY`, `EMAIL`

### Airflow variables
- `databricks_host`
- `databricks_http_path`
- `databricks_token`
- `embedding_model_name`
- `embedding_model_path`
- `embedding_dimension`
- `embedding_model_hit_rate`
- `generation_model_name`
- `generation_model_score`

---

## Naming conventions

The repository currently follows these conventions:

- Python files: `snake_case.py`
- Python functions: `snake_case`
- Python classes: `PascalCase`
- Constants: `ALL_CAPS`
- Databricks job names: `{description}_pipeline`
- Airflow DAG IDs: `snake_case`
- Airflow task IDs: `snake_case`
- Vector Search endpoint: `rag_pipeline_endpoint`
- Vector Search index: `rag_pipeline.silver.chunk_index`
- Chunk IDs: `${pmid}_chunk_${chunk_index}`

---

## Where to look first

- Main ingestion notebook: `databricks_notebooks/pubmed_to_databricks.py`
- Main chunking notebook: `databricks_notebooks/abstracts_to_chunks.py`
- Main embedding notebook: `databricks_notebooks/chunks_to_embeddings.py`
- Main query layer: `databricks_notebooks/rag_query.py`
- Streamlit UI: `databricks_notebooks/streamlit_app.py`
- Evaluation notebooks: `model_testing_notebooks/`
- Airflow orchestration: `airflow/dags/`
- Databricks job definitions: `databricks_jobs/`

---

## Operational handoff notes for future agents

The repository is intentionally notebook-first and job-first, so the fastest way to understand or change behavior is to follow the pipeline stage by stage:

1. Start with `setup.sh` and the local `.env` file to confirm the runtime environment.
2. Read the relevant notebook for the stage you are changing.
3. Check the matching Databricks job definition in `databricks_jobs/`.
4. If the change affects orchestration, inspect the corresponding Airflow DAG in `airflow/dags/`.

### Important assumptions

- The Databricks catalog is expected to be `rag_pipeline`.
- The production model selection is controlled by the production config table and by the Airflow variables used by the DAGs.
- The embedding dimension must remain consistent between the model, the embedding table, and the Vector Search index.
- The Databricks job names used by Airflow must match the job names created in the workspace.
- The model artifact path under `/Volumes/rag_pipeline/silver/models/...` must exist and be accessible.

### Common gotchas

- References to a top-level `pipelines/` package are historical; the active implementation in this workspace is notebook- and job-driven.
- `dbt/` exists but is not yet populated with implementation files in this checkout.
- The Streamlit app uses the Spark-less query path, while the notebook-based RAG flow uses the full Spark-enabled query layer.
- The query layer should read the current model selection from production config rather than relying on hard-coded defaults.

### Practical local run order

- Local environment bootstrap: `./setup.sh`
- Streamlit UI: `cd databricks_notebooks && source ~/rag_pipeline/.env && streamlit run streamlit_app.py --server.headless true`
- Airflow orchestration: `export AIRFLOW_HOME=~/rag_pipeline/airflow && airflow standalone`

These notes are meant to reduce the amount of context that a future agent has to reconstruct from scratch.

---

## Known repository notes

- The repository contains both a notebook-based RAG implementation and a more advanced Streamlit experience in `streamlit_app.py`.
- The current docs and code no longer match the older `pipelines/` layout; this document reflects the files that actually exist in the workspace.
- `dbt/` is present but not yet populated with model files in this checkout.
