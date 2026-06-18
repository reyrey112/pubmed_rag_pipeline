import os, getpass
from transformers import pipeline
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

config = spark.sql("""
    SELECT * FROM rag_pipeline.silver.production_config
    ORDER BY config_version DESC LIMIT 1
""").collect()[0]

# avoid lock from trying to access the same file
cache_dir = f"/tmp/hf_cache_{getpass.getuser()}"

os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

EMBED_MODEL_PATH = config["embedding_model_path"]
EMBEDDING_DIM = config["embedding_dimension"]
GEN_MODEL_NAME = config["generation_model_name"]

ENDPOINT_NAME = os.environ.get("VECTOR_SEARCH_ENDPOINT", "rag_pipeline_endpoint")
INDEX_NAME = os.environ.get("VECTOR_SEARCH_INDEX", "rag_pipeline.silver.chunk_index")
GEN_MODEL_NAME = os.environ.get("GEN_MODEL_NAME", "google/flan-t5-base")

# Load models once at module level
_embed_model = None
_model = None
_tokenizer = None
_vsc = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    return _embed_model


def get_model_and_tokenizer():
    """Explicitly loads model and tokenizer to replace legacy text2text-generation pipeline."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    return _model, _tokenizer


def get_vsc():
    global _vsc
    if _vsc is None:
        _vsc = VectorSearchClient()
    return _vsc


def retrieve_chunks(query, num_results=5):
    vsc = get_vsc()
    index = vsc.get_index("rag_pipeline_endpoint", "rag_pipeline.silver.chunk_index")

    embed_model = get_embed_model()
    query_vector = embed_model.encode(query).tolist()

    results = index.similarity_search(
        query_vector=query_vector,
        columns=["chunk_id", "pmid", "chunk"],
        num_results=num_results,
    )
    return results["result"]["data_array"]


def generate_answer(query, chunks):
    context = "\n\n".join([c[2] for c in chunks])  # chunk text column

    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}
Answer:"""

    model, tokenizer = get_model_and_tokenizer()
    
    # Tensor format mapping to the current active device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    # clean generated answer tokens 
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def rag_query(query):
    chunks = retrieve_chunks(query)
    answer = generate_answer(query, chunks)

    sources = list(set([c[1] for c in chunks]))  # pmid metadata column

    return {"answer": answer, "sources": sources, "retrieved_chunks": chunks}


if __name__ == "__main__":
    result = rag_query("What factors reduce viscosity in protein formulations?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s)
