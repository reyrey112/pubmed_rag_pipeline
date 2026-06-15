from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession
import time, argparse

spark = SparkSession.builder.getOrCreate()


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
    existing_indexes = [
        i["name"] for i in vsc.list_indexes(endpoint_name).get("vector_indexes", [])
    ]

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
            pipeline_type="TRIGGERED",
        )


def main():
    parser = argparse.ArgumentParser(description="Create or sync a Vector Search index")
    parser.add_argument("--source_table", default="rag_pipeline.silver.embeddings")
    parser.add_argument("--index_name", default="rag_pipeline.silver.chunk_index")
    parser.add_argument("--endpoint_name", default="rag_pipeline_endpoint")
    parser.add_argument("--embedding_dim", required=True, type=int)

    args = parser.parse_args()

    vsc = VectorSearchClient()
    endpoint_exists(vsc, args.endpoint_name)
    index_exists(
        vsc, args.endpoint_name, args.index_name, args.source_table, args.embedding_dim
    )

    print("Done.")


if __name__ == "__main__":

    main()
