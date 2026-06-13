from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer

vsc = VectorSearchClient()
index = vsc.get_index("rag_pipeline_endpoint", "rag_pipeline.silver.chunk_index")

MODEL_PATH = "/Volumes/rag_pipeline/silver/models/sentence_transformer_all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_PATH)

query = "viscosity reduction in protein formulations"
query_vector = model.encode(query).tolist()

results = index.similarity_search(
    query_vector=query_vector,
    columns=["chunk_id", "pmid", "chunk"],
    num_results=5
)

for r in results["result"]["data_array"]:
    print(r)