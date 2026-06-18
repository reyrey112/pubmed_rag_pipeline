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
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "path": "/Volumes/rag_pipeline/silver/models/all-MiniLM-L6-v2",
        "dim": 384,
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "path": "/Volumes/rag_pipeline/silver/models/all-mpnet-base-v2",
        "dim": 768,
    },
    {
        "name": "allenai/specter2_base",
        "path": "/Volumes/rag_pipeline/silver/models/specter2_base",
        "dim": 768,
    },
]
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
    model, df_chunks: pd.DataFrame, df_eval: pd.DataFrame, k: int = 5
) -> dict:

    print("Embedding Chunks")
    chunk_embeddings = embed_texts(model, df_chunks["chunk"].tolist())
    chunk_ids = df_chunks["chunk_id"].tolist()

    print("Embedding Questions")

    question_embeddings = embed_texts(model, df_eval["question"].tolist())

    hits = 0
    reciprocal_ranks = []

    for i, row in df_eval.iterrows():
        q_vec = question_embeddings[i]
        correct_id = row["correct_chunk_id"]

        # Cosine similarity
        norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_vec)
        similarities = chunk_embeddings @ q_vec / norms
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_ids = [chunk_ids[idx] for idx in top_k_indices]

        if correct_id in top_k_ids:
            hits += 1
            rank = top_k_ids.index(correct_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return {
        f"hit_rate_at_{k}": hits / len(df_eval),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "num_eval_pairs": len(df_eval),
        "k": k,
    }


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

            try:
                model = SentenceTransformer(model_name)
                metrics = compute_metrics(model, df_chunks, df_eval, k=5)

                mlflow.log_metrics(metrics)
                print(f"  Hit Rate@5: {metrics['hit_rate_at_5']:.3f}")
                print(f"  MRR:        {metrics['mrr']:.3f}")

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

    # Summary table
    df_results = pd.DataFrame(all_results)
    print("\n=== RESULTS SUMMARY ===")
    print(df_results.sort_values("hit_rate_at_5", ascending=False).to_string(index=False))

    # Save results to Delta
    spark_df = spark.createDataFrame(df_results).withColumn(
        "evaluated_at", current_timestamp()
    )

    spark_df.write.format("delta").mode("append").saveAsTable(
        "rag_pipeline.silver.embedding_eval_results"
    )

    # Return best model
    best = df_results.loc[df_results["hit_rate_at_5"].idxmax(), "model"]
    print(f"\nBest model: {best}")
    return best


if __name__ == "__main__":
    run_evaluation()
