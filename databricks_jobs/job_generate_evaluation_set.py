import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

w = WorkspaceClient()


def create_job():
    job = w.jobs.create(
        name="generate_evaluation_set_pipeline",
        tasks=[
            jobs.Task(
                task_key="generate_eval_set",
                spark_python_task=jobs.SparkPythonTask(
                    python_file="/Workspace/Users/reydencdavies@gmail.com/rag_pipeline/model_testing_notebooks/generate_evaluation_set.py"
                ),
                environment_key="Serverless",
            )
        ],
        environments=[
            jobs.JobEnvironment(
                environment_key="Serverless",
                spec=compute.Environment(
                    client="2", dependencies=["transformers", "torch", "google-genai"]
                ),
            )
        ],
    )
    return job


# Create the job
existing_jobs = w.jobs.list(name="generate_evaluation_set_pipeline")
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
