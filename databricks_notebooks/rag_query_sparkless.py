import os, getpass, sys
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
util_path = os.path.join(repo_root, "airflow", "dags", "util")
if util_path not in sys.path:
    sys.path.append(util_path)

from conversation_history import enrich_query
from iterative_retrieval import iterative_retrieve
from gemini_call import gemini_call
from interview_state import (
    build_retrieval_query,
    build_generation_context,
    InterviewState,
)
from databricks import sql
import os

from dotenv import load_dotenv

load_dotenv()

_gemini_client = None

def _parse_section(text: str, tag: str) -> str:
    """
    Extracts content between <tag> and </tag> from Claude's response.
    Returns empty string if tag not found.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    start = text.find(open_tag)
    end = text.find(close_tag)

    if start == -1 or end == -1:
        return ""

    return text[start + len(open_tag):end].strip()

def _get_gemini_client():

    global _gemini_client
    if _gemini_client is None:
        from google import genai

        try:
            api_key = dbutils.secrets.get(scope="rag_pipeline", key="GEMINI_API_KEY")

        except NameError as e:
            api_key = os.environ.get("GEMINI_API_KEY")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


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


def generate_report(
    question: str,
    generation_context: str,
    chunks: list,
) -> dict:
    """
    Generates a structured experiment design report using Claude,
    grounded in retrieved literature chunks.

    Parameters:
        question            — original user research question
        generation_context  — structured context from build_generation_context()
        chunks              — retrieved chunks from iterative_retrieve()

    Returns:
        dict with one key per report section, plus raw_response for debugging
    """
    # Format chunks for the prompt
    chunks_text = "\n\n".join(
        [f"[Chunk {i+1} | PMID: {c[1]}]\n{c[2]}" for i, c in enumerate(chunks)]
    )

    prompt = f"""You are an expert scientific research advisor helping design an experiment.
Use ONLY the provided research context and literature excerpts to inform your recommendations.
Where the literature is silent, say so clearly rather than speculating.

{generation_context}

RETRIEVED LITERATURE:
{chunks_text}

Generate a structured experiment design report with exactly these sections.
Use the XML tags exactly as shown:

<background>
What does the retrieved literature say about this problem? Summarize key findings relevant to the research question. 2-3 paragraphs.
</background>

<approach>
What experimental approach do you recommend based on the literature? Be specific about methods, techniques, and sequence of steps. Reference the literature where relevant.
</approach>

<parameters>
What are the key process parameters, their recommended ranges or values, and the rationale from the literature? Use a structured list.
</parameters>

<controls>
What positive and negative controls are needed? What are the success criteria and how should results be interpreted?
</controls>

<risks>
What are the known failure modes, risks, or pitfalls identified in the literature? What mitigation strategies are recommended?
</risks>

<gaps>
What aspects of the research question are NOT covered by the retrieved literature? What additional experiments or literature review would strengthen the design?
</gaps>

<references>
Which literature excerpts (by chunk number and PMID) informed each section of this report?
</references>"""

    client = _get_gemini_client()

    try:
        response = gemini_call(client, prompt, max_output_tokens=10000)

        print(f"response: {response.text}")

        raw = response.text.strip()

        # add trying in cause returns markdown strings
        # import json
        # try:
        #     report = json.loads(response.text.strip())["report"]
        # except KeyError as e:
        #     report = json.loads(response.text.strip())

        # sections = {
        #     "background": report["background"],
        #     "approach": report["approach"],
        #     "parameters": report["parameters"],
        #     "controls": report["controls"],
        #     "risks": report["risks"],
        #     "gaps": report["gaps"],
        #     "references": report["references"],
        #     "raw_response": raw,  # keep for debugging
        # }
        sections = {
            "background":  _parse_section(raw, "background"),
            "approach":    _parse_section(raw, "approach"),
            "parameters":  _parse_section(raw, "parameters"),
            "controls":    _parse_section(raw, "controls"),
            "risks":       _parse_section(raw, "risks"),
            "gaps":        _parse_section(raw, "gaps"),
            "references":  _parse_section(raw, "references"),
            "raw_response": raw,  # keep for debugging
        }
        # Warn if any sections failed to parse
        missing = [k for k, v in sections.items() if not v and k != "raw_response"]
        if missing:
            print(f"Warning: sections missing from response: {missing}")

        print(f"raw: {raw}")

        return sections

    except Exception as e:
        print(f"Report generation failed: {e}")
        return {
            "background": f"Report generation failed: {e}",
            "approach": "",
            "parameters": "",
            "controls": "",
            "risks": "",
            "gaps": "",
            "references": "",
            "raw_response": "",
        }


def rag_query(
    query: str,
    history: list[dict] = None,
    status_callback=None,
) -> dict:
    """
    Main RAG query function with iterative retrieval and conversation memory.

    Parameters:
        query            — raw user message
        history          — conversation history for query enrichment
        status_callback  — optional callable(message: str) for live UI updates
                           passed through to iterative_retrieve()

    Returns:
        answer           — generated answer string
        sources          — list of source PMIDs
        retrieved_chunks — full chunk data from all iterations
        query_used       — initial enriched query (first iteration)
        chunk_ids        — all unique chunk IDs retrieved
        iterations       — how many retrieval loops ran
        queries_used     — all queries used across iterations
        sufficient       — whether sufficient context was found
        final_reasoning  — Claude's reasoning on final sufficiency assessment
    """
    retrieved_chunks = []
    query_used = enrich_query(query, history or [])

    retrieval_result = iterative_retrieve(
        question=query,
        initial_query=query_used,
        retrieve_fn=retrieve_chunks,
        status_callback=status_callback,
    )

    chunks = retrieval_result["chunks"]
    queries_used = retrieval_result["queries_used"]
    iterations = retrieval_result["iterations"]
    sufficient = retrieval_result["sufficient"]
    final_reasoning = retrieval_result["final_reasoning"]

    answer = generate_answer(query, chunks)

    sources = list(set([c[1] for c in chunks]))
    chunk_ids = [c[0] for c in chunks]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
        "query_used": query_used,
        "chunk_ids": chunk_ids,
        "iterations": iterations,
        "queries_used": queries_used,
        "sufficient": sufficient,
        "final_reasoning": final_reasoning,
    }

def experiment_design_query(
    state: InterviewState,
    status_callback=None,
) -> dict:
    """
    Full experiment design pipeline — uses interview answers to
    build a targeted retrieval query, runs iterative retrieval,
    and generates a structured experiment design report.

    Parameters:
        state            — completed InterviewState with all answers
        status_callback  — optional callable for live UI status updates

    Returns:
        report           — dict of report sections (background, approach, etc.)
        chunks           — all retrieved chunks
        chunk_ids        — list of chunk IDs for history writer
        queries_used     — all queries used across retrieval iterations
        iterations       — number of retrieval loops
        sufficient       — whether sufficient context was found
        retrieval_query  — the query used for initial retrieval
    """
    # Step 1 — build retrieval query from interview answers
    retrieval_query = build_retrieval_query(state)

    if status_callback:
        status_callback(f"📋 Built retrieval query from your research context")
        status_callback(f"🔍 Query: {retrieval_query[:100]}...")

    # Step 2 — iterative retrieval using interview-informed query
    retrieval_result = iterative_retrieve(
        question=state.original_question,
        initial_query=retrieval_query,
        retrieve_fn=retrieve_chunks,
        status_callback=status_callback,
    )

    chunks = retrieval_result["chunks"]
    queries_used = retrieval_result["queries_used"]
    iterations = retrieval_result["iterations"]
    sufficient = retrieval_result["sufficient"]

    if status_callback:
        status_callback(f"📝 Generating experiment design report...")

    # Step 3 — build structured generation context
    generation_context = build_generation_context(state)

    # Step 4 — generate structured report
    report = generate_report(
        question=state.original_question,
        generation_context=generation_context,
        chunks=chunks,
    )

    # Step 5 — extract metadata
    chunk_ids = [c[0] for c in chunks]

    return {
        "report": report,
        "chunks": chunks,
        "chunk_ids": chunk_ids,
        "queries_used": queries_used,
        "iterations": iterations,
        "sufficient": sufficient,
        "retrieval_query": retrieval_query,
    }

if __name__ == "__main__":
    result = rag_query("What factors reduce viscosity in protein formulations?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s)
