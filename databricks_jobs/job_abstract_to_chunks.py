from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()

job = w.jobs.create(
    name="abstract_chunking_pipeline",
    tasks=[
        jobs.Task(
            task_key="chunk_abstracts",
            notebook_task=jobs.NotebookTask(
                notebook_path="Workspace/Users/reydencdavies@gmail.com/rag_pipeline/test" 
            ),
            environment_key="Serverless"
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="Serverless",
            spec=jobs.EnvironmentSpec(
                client="1" 
            )
        )
    ]
)