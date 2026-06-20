import os, getpass, sys
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer

# 1. Get the absolute path of the folder containing rag_query.py (databricks_notebooks)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Move up one level to the 'rag_pipeline' root directory
repo_root = os.path.abspath(os.path.join(current_dir, ".."))

# 3. Path to the targeted util folder
util_path = os.path.join(repo_root, "airflow", "dags", "util")

# 4. Inject it into Python's search path
if util_path not in sys.path:
    sys.path.append(util_path)

# 5. Import your module cleanly
from conversation_history import enrich_query

from databricks import sql
import os


def get_latest_config():
    conn = sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM rag_pipeline.silver.production_config
        ORDER BY config_version DESC LIMIT 1
    """)
    row = cursor.fetchone()
    columns = [d[0] for d in cursor.description]
    cursor.close()
    conn.close()
    return dict(zip(columns, row))


_config = None


def get_config():
    global _config
    if _config is None:
        _config = get_latest_config()  # SQL connector call, runs once
    return _config


config = get_config()

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
EMBED_MODEL_NAME = config["embedding_model_name"]

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
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
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


def rag_query(query: str, history: list[dict] = None) -> dict:
    """
    Main RAG query function.

    Parameters:
        query    — raw user message
        history  — list of { role, content } dicts from conversation history
                   if None or empty, query is used as-is (no enrichment)

    Returns:
        answer          — generated answer string
        sources         — list of source PMIDs
        retrieved_chunks— full chunk data
        query_used      — enriched query sent to retrieval (for history writer)
        chunk_ids       — list of chunk_ids retrieved (for history writer)
    """

    query_used = enrich_query(query, history or [])
    chunks = retrieve_chunks(query_used)
    answer = generate_answer(query_used, chunks)
    sources = list(set([c[1] for c in chunks]))
    chunk_ids = [c[0] for c in chunks]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
        "query_used": query_used,  
        "chunk_ids": chunk_ids,  
    }


if __name__ == "__main__":
    result = rag_query("What factors reduce viscosity in protein formulations?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s)
