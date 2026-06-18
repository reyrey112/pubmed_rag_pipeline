from dotenv import load_dotenv

load_dotenv()

# Step 2: Upload CSVs to databricks
from databricks.sdk import WorkspaceClient
import os


def upload_csv_to_volume():

    db_workspace_conn = WorkspaceClient(
        host=os.environ.get("DATABRICKS_HOST"), token=os.environ.get("DATABRICKS_TOKEN")
    )

    csv_dir = os.environ.get("OUTPUT_DIR")
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            local_path = os.path.join(csv_dir, filename)
            dbricks_path = f"{os.environ.get('DATABRICKS_VOLUME_PATH')}/{filename}"

            with open(local_path, "rb") as f:
                db_workspace_conn.files.upload(dbricks_path, f, overwrite=True)
            print(f" Uploaded: {filename}")

    print("\n Done")


if __name__ == "__main__":
    upload_csv_to_volume()
