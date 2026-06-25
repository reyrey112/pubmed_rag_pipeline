# util/iterative_retrieval.py

import json
import os
from google import genai
from dotenv import load_dotenv
from databricks import sql

load_dotenv()
from google.genai import types

MAX_TOTAL_CHUNKS = 10
MAX_ITERATIONS = 3
MODEL = "gemini-2.5-flash"

SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]
_gemini_client = None


def _get_gemini_client():
    from google import genai

    global _gemini_client
    if _gemini_client is None:
        try:
            api_key = dbutils.secrets.get(scope="rag_pipeline", key="GEMINI_API_KEY")

        except NameError as e:
            api_key = os.environ.get("GEMINI_API_KEY")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


_config = None


def get_config():
    global _config
    if _config is None:
        _config = get_latest_config()  # SQL connector call, runs once
    return _config


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


config = get_latest_config()

# GEN_MODEL_NAME = config[""]


def assess_sufficiency(question: str, chunks: list, model=MODEL) -> dict:
    """
    Asks Gemini whether the retrieved chunks provide sufficient context
    to answer the question confidently.

    Parameters:
        question  — original user question (not enriched query)
        chunks    — list of retrieved chunks [[chunk_id, pmid, chunk_text], ...]

    Returns:
        {
            "sufficient": bool,
            "missing_aspect": str or None,  — what's missing if not sufficient
            "reasoning": str                — brief explanation
        }
    """

    from gemini_call import gemini_call

    if not chunks:
        return {
            "sufficient": False,
            "missing_aspect": "No chunks were retrieved",
            "reasoning": "Empty retrieval result",
        }

    # Format chunks for the prompt
    chunks_text = "\n\n".join(
        [
            f"Chunk {i+1}:\n{str(c[2])[:300]}"  # explicitly cast to str
            for i, c in enumerate(chunks)
        ]
    )

    prompt = f"""You are evaluating whether retrieved research paper excerpts provide sufficient context to answer a question.

Question: {question}

Retrieved excerpts:
{chunks_text}

Evaluate whether these excerpts provide sufficient context to answer the question. Consider:
1. Are at least 2-3 excerpts directly relevant to the question?
2. Do the excerpts contain specific enough detail (not just tangentially related)?
3. Is there an obvious sub-topic the question implies that the excerpts don't cover?

Respond with ONLY a JSON object, no other text:
{{
    "sufficient": true or false,
    "missing_aspect": "brief description of what is missing, or null if sufficient",
    "reasoning": "one sentence explanation"
}}"""

    client = _get_gemini_client()

    try:

        response = gemini_call(
            client,
            prompt,
            response_mime_type="application/json",
            safety_settings=SAFETY_SETTINGS,
        )

        text = response.text

        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        result = json.loads(text)

        # Validate expected keys exist
        if "sufficient" not in result:
            raise ValueError("Missing 'sufficient' key in response")

        return {
            "sufficient": bool(result["sufficient"]),
            "missing_aspect": result.get("missing_aspect"),
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as e:
        print(f"Sufficiency assessment failed: {e} — defaulting to sufficient=True")
        return {
            "sufficient": True,
            "missing_aspect": None,
            "reasoning": f"Assessment failed: {e}",
        }


def reformulate_query(
    original_question: str,
    missing_aspect: str,
    previous_queries: list[str],
    chunks_so_far: list,
) -> str:
    """
    Generates a new targeted search query focused on the missing aspect.

    Parameters:
        original_question  — the user's original question
        missing_aspect     — what assess_sufficiency() said was missing
        previous_queries   — all queries tried so far (to avoid repetition)
        chunks_so_far      — chunks already retrieved (for context)

    Returns:
        A new search query string targeting the missing aspect
    """
    previous_queries_text = (
        "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
    )

    chunks_summary = (
        "\n".join([f"- {c[2][:150]}..." for c in chunks_so_far[:3]])
        if chunks_so_far
        else "None"
    )

    prompt = f"""You are helping refine a scientific literature search query.

Original question: {original_question}

What is missing from current search results: {missing_aspect}

Queries already tried (do NOT generate something similar to these):
{previous_queries_text}

Sample of what has already been retrieved:
{chunks_summary}

Generate ONE new search query that:
1. Specifically targets the missing aspect
2. Is clearly different from the queries already tried
3. Is phrased as a natural language sentence (not keywords)
4. Stays grounded in the original question's domain
5. Is concise — 1-2 sentences maximum

Respond with ONLY the query text, nothing else."""

    client = _get_gemini_client()

    try:
        response: types.GenerateContentResponse
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                response_mime_type="application/json",
                safety_settings=SAFETY_SETTINGS,
                # system_instruction="You are a strict LLM evaluation judge. Respond with ONLY a valid raw JSON object, no markdown code fences, and no other text.",
            ),
        )
        text = response.text

        return text

    except Exception as e:
        print(f"Query reformulation failed: {e} — falling back to original question")
        return original_question


def deduplicate_chunks(
    existing_chunks: list,
    new_chunks: list,
) -> list:
    """
    Merges two lists of chunks, removing duplicates by chunk_id.
    Preserves order — existing chunks first, new chunks appended after.
    Caps total at MAX_TOTAL_CHUNKS.

    Parameters:
        existing_chunks — chunks from previous iterations
        new_chunks      — chunks from current iteration

    Returns:
        Merged deduplicated list capped at MAX_TOTAL_CHUNKS
    """
    seen_ids = {c[0] for c in existing_chunks}  # set of chunk_ids already seen
    merged = list(existing_chunks)  # start with existing

    added = 0
    for chunk in new_chunks:
        chunk_id = chunk[0]
        if chunk_id not in seen_ids:
            merged.append(chunk)
            seen_ids.add(chunk_id)
            added += 1

    print(
        f"Deduplication: {len(existing_chunks)} existing + "
        f"{len(new_chunks)} new → {added} added → "
        f"{len(merged)} total (cap={MAX_TOTAL_CHUNKS})"
    )

    return merged[:MAX_TOTAL_CHUNKS]


def iterative_retrieve(
    question: str,
    initial_query: str,
    retrieve_fn,
    max_iterations: int = MAX_ITERATIONS,
    status_callback=None,
) -> dict:
    """
    Iteratively retrieves chunks until sufficient context is found
    or max_iterations is reached.

    Parameters:
        question         — original user question (used for sufficiency assessment)
        initial_query    — enriched query from conversation history (used for first retrieval)
        retrieve_fn      — callable that takes a query string and returns chunks
                           (passed in to keep this module decoupled from rag_query.py)
        max_iterations   — maximum number of retrieval loops (default 3)
        status_callback  — optional callable(message: str) for UI status updates
                           called between iterations so Streamlit can show progress

    Returns:
        {
            "chunks":           list of deduplicated chunks,
            "queries_used":     list of all queries tried,
            "iterations":       int, how many loops ran,
            "sufficient":       bool, whether sufficiency was reached,
            "final_reasoning":  str, Claude's reasoning on final assessment,
        }
    """

    def notify(message: str):
        print(message)
        if status_callback:
            status_callback(message)

    all_chunks = []
    queries_used = []
    current_query = initial_query

    for iteration in range(1, max_iterations + 1):
        notify(f"🔍 Search {iteration}/{max_iterations}: {current_query[:80]}...")

        # Retrieve chunks for current query
        new_chunks = retrieve_fn(current_query)

        # Deduplicate against what we already have
        before_count = len(all_chunks)
        all_chunks = deduplicate_chunks(all_chunks, new_chunks)
        after_count = len(all_chunks)
        queries_used.append(current_query)

        new_unique = after_count - before_count
        notify(f"   Retrieved {len(new_chunks)} chunks, {new_unique} new unique")

        # Stopping condition 3 — no new unique chunks (diminishing returns)
        if iteration > 1 and new_unique == 0:
            notify("   No new unique chunks found — stopping early")
            return {
                "chunks": all_chunks,
                "queries_used": queries_used,
                "iterations": iteration,
                "sufficient": True,  # treat as sufficient — more searching won't help
                "final_reasoning": "Stopped early — no new unique chunks found",
            }

        # Assess sufficiency with current accumulated chunks
        notify(f"   Assessing sufficiency of {len(all_chunks)} chunks...")
        assessment = assess_sufficiency(question, all_chunks)
        notify(f"   Sufficient: {assessment['sufficient']} — {assessment['reasoning']}")

        # Stopping condition 1 — sufficient context found
        if assessment["sufficient"]:
            notify(f"✅ Sufficient context found after {iteration} iteration(s)")
            return {
                "chunks": all_chunks,
                "queries_used": queries_used,
                "iterations": iteration,
                "sufficient": True,
                "final_reasoning": assessment["reasoning"],
            }

        # Stopping condition 2 — max iterations reached
        if iteration == max_iterations:
            notify(
                f"⚠️ Max iterations ({max_iterations}) reached. generating with available context"
            )
            return {
                "chunks": all_chunks,
                "queries_used": queries_used,
                "iterations": iteration,
                "sufficient": False,
                "final_reasoning": assessment["reasoning"],
            }

        # Not sufficient and iterations remain -> reformulate query
        notify(f"   Missing: {assessment['missing_aspect']}")
        current_query = reformulate_query(
            original_question=question,
            missing_aspect=assessment["missing_aspect"],
            previous_queries=queries_used,
            chunks_so_far=all_chunks,
        )
        notify(f"   Reformulated query: {current_query[:80]}...")

    # Should never reach here but return safely if it does
    return {
        "chunks": all_chunks,
        "queries_used": queries_used,
        "iterations": max_iterations,
        "sufficient": False,
        "final_reasoning": "Loop exited unexpectedly",
    }
