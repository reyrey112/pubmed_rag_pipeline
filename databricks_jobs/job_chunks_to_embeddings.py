import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

w = WorkspaceClient()


def create_job():
    job = w.jobs.create(
        name="chunks_to_embeddings_pipeline",
        tasks=[
            jobs.Task(
                task_key="embed_chunks",
                spark_python_task=jobs.SparkPythonTask(
                    python_file="/Workspace/Users/reydencdavies@gmail.com/rag_pipeline/databricks_notebooks/chunks_to_embeddings.py"
                ),
                environment_key="Serverless",
            )
        ],
        environments=[
            jobs.JobEnvironment(
                environment_key="Serverless",
                spec=compute.Environment(
                    client="2", dependencies=["sentence-transformers==2.7.0", "torch==2.2.0"]
                ),
            )
        ],
    )
    return job


# Create the job
existing_jobs = w.jobs.list(name="chunks_to_embeddings_pipeline")
existing = next(iter(existing_jobs), None)

if existing:
    job_id = existing.job_id
    print(f"Using existing job: {job_id}")
else:
    job = create_job()
    job_id = job.job_id
    print(f"Created new job: {job_id}")


# Run it
run = w.jobs.run_now(job_id=job_id)
print(f"Started run ID: {run.run_id}")

