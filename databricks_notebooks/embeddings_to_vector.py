from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession
import time

spark = SparkSession.builder.getOrCreate()

SOURCE_TABLE = "rag_pipeline.silver.embeddings"
INDEX_NAME = "rag_pipeline.silver.chunk_index"
ENDPOINT_NAME = "rag_pipeline_endpoint"
EMBEDDING_DIM = 384

def endpoint_exists(vsc, endpoint_name):
    existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if endpoint_name not in existing:
        print(f"Creating endpoint {endpoint_name}...")
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")

        # wait for it to come online
        while True:
            status = vsc.get_endpoint(endpoint_name)
            state = status.get("endpoint_status", {}).get("state")
            print(f"Endpoint state: {state}")
            if state == "ONLINE":
                break
            time.sleep(30)
    else:
        print(f"Endpoint {endpoint_name} already exists")

def index_exists(vsc, endpoint_name, index_name, source_table, embedding_dim):
    existing_indexes = [i["name"] for i in vsc.list_indexes(endpoint_name).get("vector_indexes", [])]

    if index_name in existing_indexes:
        print(f"Index {index_name} already exists, triggering sync...")
        vsc.get_index(endpoint_name, index_name).sync()
    else:
        print(f"Creating index {index_name}...")
        vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=source_table,
            index_name=index_name,
            primary_key="chunk_id",
            embedding_dimension=embedding_dim,
            embedding_vector_column="embedding",
            pipeline_type="TRIGGERED"
        )


if __name__ == "__main__":

    vsc = VectorSearchClient()
    endpoint_exists(vsc, ENDPOINT_NAME)
    index_exists(vsc, ENDPOINT_NAME, INDEX_NAME, SOURCE_TABLE, EMBEDDING_DIM)

    print("Done.")