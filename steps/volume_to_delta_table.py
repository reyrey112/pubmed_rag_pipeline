from dotenv import load_dotenv

load_dotenv()

# Convert Databricks Volumes to Delta Tables
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from databricks import sql
import os, time


class Conversion:
    def __init__(self) -> None:
        self.connect()
        self.catalog = os.environ.get("DATABRICKS_CATALOG")
        self.schema = os.environ.get("DATABRICKS_SCHEMA")

    def connect(self):
        self.conn = WorkspaceClient(
            host=os.environ.get("DATABRICKS_HOST"),
            token=os.environ.get("DATABRICKS_TOKEN"),
        )

    def run_sql(self, statement):
        response = self.conn.statement_execution.execute_statement(
            warehouse_id=os.environ.get("DATABRICKS_WAREHOUSE_ID"),
            statement=statement,
        )

        while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(2)
            response = self.conn.statement_execution.get_statement(
                response.statement_id
            )
        return response

    def volume_to_delta_table(self):

        files = self.conn.dbfs.list(os.environ.get("DATABRICKS_VOLUME_PATH"))
        csv_files = [f.path for f in files if f.path.endswith(".csv")]

        for filepath in csv_files:
            table_name = filepath.split("/")[-1].replace(".csv", "")
            full_table_name = f"{self.catalog}.{self.schema}.{table_name}"

            query = f"""
                    CREATE OR REPLACE TABLE {full_table_name}
                    USING DELTA
                    AS SELECT * FROM read_files(
                    '{filepath}',
                    format => 'csv',
                    header => true,
                    inferSchema => true
                    )
                    """
            print(f"Creating {full_table_name}")
            result = self.run_sql(query)

            if result.status.state == StatementState.SUCCEEDED:
                print(f"Done Creating {full_table_name}")

            else:
                print(f"Failed with: {result.status.error}")
        print("Done creating tables")


if __name__ == "__main__":
    new_conversion = Conversion()

    new_conversion.volume_to_delta_table()
