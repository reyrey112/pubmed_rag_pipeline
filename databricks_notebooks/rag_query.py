import os 
from transformers import pipeline
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer


os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

EMBED_MODEL_PATH = "/Volumes/rag_pipeline/silver/models/sentence_transformer_all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"

def retrieve_chunks(query, num_results=5):
    vsc = VectorSearchClient()
    index = vsc.get_index("rag_pipeline_endpoint", "rag_pipeline.silver.chunk_index")

    embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    query_vector = embed_model.encode(query).tolist()

    results = index.similarity_search(
        query_vector=query_vector,
        columns=["chunk_id", "pmid", "chunk"],
        num_results=num_results
    )
    return results["result"]["data_array"]

def generate_answer(query, chunks):
    context = "\n\n".join([c[2] for c in chunks])  # chunk text column

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}
Answer:"""

    generator = pipeline("text2text-generation", model=GEN_MODEL_NAME)
    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]

def rag_query(query):
    chunks = retrieve_chunks(query)
    answer = generate_answer(query, chunks)

    sources = list(set([c[1] for c in chunks]))  # title column, deduped

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks
    }

if __name__ == "__main__":
    result = rag_query("What factors reduce viscosity in protein formulations?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s)