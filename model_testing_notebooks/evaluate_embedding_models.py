import numpy as np
import pandas as pd
import mlflow
import os, getpass
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import current_timestamp

spark = SparkSession.builder.getOrCreate()

# avoid lock from trying to access the same file
cache_dir = f"/tmp/hf_cache_{getpass.getuser()}"

os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

MODELS_TO_EVALUATE = [
    {
        "name": "all-MiniLM-L6-v2",
        "path": "/Volumes/rag_pipeline/silver/models/all-MiniLM-L6-v2",
        "dim": 384,
    },
    {
        "name": "all-mpnet-base-v2",
        "path": "/Volumes/rag_pipeline/silver/models/all-mpnet-base-v2",
        "dim": 768,
    },
    {
        "name": "specter2_base",
        "path": "/Volumes/rag_pipeline/silver/models/specter2_base",
        "dim": 768,
    },
]

K_VALUES = [5, 10]

MLFLOW_EXPERIMENT = "/Users/reydencdavies@gmail.com/embedding_model_evaluation"

CHUNKS_TABLE = "rag_pipeline.silver.chunks"
EVAL_TABLE = "rag_pipeline.silver.eval_questions"


def load_data():
    df_chunks = spark.table(CHUNKS_TABLE).select("chunk_id", "chunk").toPandas()

    df_eval = (
        spark.table(EVAL_TABLE)
        .select("question", "chunk_id")
        .toPandas()
        .rename(columns={"chunk_id": "correct_chunk_id"})
    )

    return df_chunks, df_eval


def embed_texts(model, texts: list, batch_size: int = 32) -> np.ndarray:
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, batch_size=batch_size)
        all_embeddings.append(embeddings)

        done = min(i + batch_size, total)
        print(f"  Embedded {done}/{total} ({done/total*100:.0f}%)")

    return np.vstack(all_embeddings)


def compute_metrics(
    model,
    df_chunks: pd.DataFrame,
    df_eval: pd.DataFrame,
    k_values: list = K_VALUES,
) -> dict:
    """
    Computes retrieval metrics at each k in k_values in a single ranking pass.

    Metrics per k:
      - hit_rate@k   : fraction of queries where the correct chunk appears in top-k.
      - mrr@k        : mean reciprocal rank, only crediting ranks within top-k.
    """

    print("Embedding Chunks")
    chunk_embeddings = embed_texts(model, df_chunks["chunk"].tolist())
    chunk_ids = df_chunks["chunk_id"].tolist()

    print("Embedding Questions")

    question_embeddings = embed_texts(model, df_eval["question"].tolist())

    hits = {k: 0 for k in k_values}
    reciprocal_ranks = {k: [] for k in k_values}
    max_k = max(k_values)

    # normalize embeddings once
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    normed_chunks = chunk_embeddings / np.where(chunk_norms == 0, 1, chunk_norms)

    for i, row in df_eval.iterrows():
        q_vec = question_embeddings[i]
        correct_id = row["correct_chunk_id"]

        # cosine similarity
        q_norm = np.linalg.norm(q_vec)
        similarities = normed_chunks @ (q_vec / (q_norm if q_norm else 1))

        top_max_k_indices = np.argsort(similarities)[::-1][:max_k]
        top_max_k_ids = [chunk_ids[idx] for idx in top_max_k_indices]

        for k in k_values:
            top_k_ids = top_max_k_ids[:k]

            if correct_id in top_k_ids:
                hits[k] += 1
                rank = top_k_ids.index(correct_id) + 1
                reciprocal_ranks[k].append(1.0 / rank)
            else:
                reciprocal_ranks[k].append(0.0)

    n = len(df_eval)
    metrics = {"num_eval_pairs": n}

    for k in k_values:
        metrics[f"hit_rate_at_{k}"] = hits[k] / n
        metrics[f"mrr_at_{k}"] = sum(reciprocal_ranks[k]) / n

    return metrics


def run_evaluation():
    from sentence_transformers import SentenceTransformer

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df_chunks, df_eval = load_data()
    print(f"Evaluating on {len(df_eval)} Q&A pairs, {len(df_chunks)} chunks")

    all_results = []

    for model_cfg in MODELS_TO_EVALUATE:
        model_name = model_cfg["name"]
        print(f"\nEvaluating: {model_name}")

        run_name = (
            f"{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("chunks_table", CHUNKS_TABLE)
            mlflow.log_param("eval_table", EVAL_TABLE)
            mlflow.log_param("k_values", str(K_VALUES))

            try:
                model = SentenceTransformer(model_name)
                metrics = compute_metrics(model, df_chunks, df_eval, k_values=K_VALUES)

                mlflow.log_metrics(metrics)

                for k in K_VALUES:
                    print(f"  --- k={k} ---")
                    print(f"  Hit Rate@{k}:  {metrics[f'hit_rate_at_{k}']:.3f}")
                    print(f"  MRR@{k}:       {metrics[f'mrr_at_{k}']:.3f}")

                all_results.append(
                    {
                        "model": model_name,
                        "model_path": model_cfg["path"],
                        "embedding_dim": model_cfg["dim"],
                        **metrics,
                    }
                )

            except Exception as e:
                print(f"  Failed: {e}")
                mlflow.log_param("error", str(e))

    df_results = pd.DataFrame(all_results)

    display_cols = ["model"] + [
        f"{m}_at_{k}" for k in K_VALUES for m in ["hit_rate", "mrr"]
    ]
    print("\n=== RESULTS SUMMARY ===")
    print(
        df_results[display_cols]
        .sort_values("hit_rate_at_5", ascending=False)
        .to_string(index=False)
    )

    spark_df = spark.createDataFrame(df_results).withColumn(
        "evaluated_at", current_timestamp()
    )
    spark_df.write.format("delta").mode("append").saveAsTable(
        "rag_pipeline.silver.embedding_eval_results"
    )

    best = df_results.loc[df_results["hit_rate_at_5"].idxmax(), "model"]
    print(f"\nBest model by Hit Rate@5: {best}")
    return best


if __name__ == "__main__":
    run_evaluation()
