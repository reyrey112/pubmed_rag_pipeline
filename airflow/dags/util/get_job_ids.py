from databricks.sdk import WorkspaceClient


def get_job_id(job_name: str) -> int:

    w = WorkspaceClient()
    for job in w.jobs.list(name=job_name):
        job_id = job.job_id
        print(job.job_id, job.settings.name)

    return job_id
    
