from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
import os
import argparse


def create_embeddings(chunks_table: str, embeddings_table: str, model_name, model_path):
    spark = SparkSession.builder.getOrCreate()

    # # Download model to shared volume (so all workers can access it)
    # MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # MODEL_PATH = (
    #     "/Volumes/rag_pipeline/silver/models/sentence_transformer_all-MiniLM-L6-v2"
    # )

    # Create directory if it doesn't exist
    # os.makedirs(MODEL_PATH, exist_ok=True)

    # Create embeddings
    @pandas_udf(ArrayType(FloatType()))
    def embed_udf(texts: pd.Series) -> pd.Series:
        import os

        os.environ["HF_HOME"] = "/tmp/hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf_cache"

        from sentence_transformers import SentenceTransformer

        config_file = os.path.join(model_path, "config.json")

        if os.path.exists(config_file):
            print("Model Already Saved, Loading model")
        else:
            print("Downloading and Saving model")
            model = SentenceTransformer(model_name)
            model.save(model_path)
            print("Model saved")

        model = SentenceTransformer(model_path)
        embeddings = model.encode(texts.tolist(), show_progress_bar=False)
        return pd.Series([e.tolist() for e in embeddings])

    df_chunks = spark.table(chunks_table)

    df_embedded = df_chunks.withColumn("embedding", embed_udf("chunk"))

    df_embedded.write.format("delta").mode("overwrite").saveAsTable(embeddings_table)
    spark.sql(
        f"ALTER TABLE {embeddings_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    print(f"Embedded {df_embedded.count()} chunks")

def main():
    parser = argparse.ArgumentParser(description="Embed chunks using a configurable model")
    parser.add_argument("--chunks_table", default=f"rag_pipeline.silver.chunks")
    parser.add_argument("--embeddings_table", default=f"rag_pipeline.silver.embeddings")
    parser.add_argument("--model_name", required=True, help="HF model name, e.g. sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model_path", required=True, help="Volume path to save/load the model")

    args = parser.parse_args()

    create_embeddings(
        chunks_table=args.chunks_table,
        embeddings_table=args.embeddings_table,
        model_name=args.model_name,
        model_path=args.model_path,
    )

if __name__ == "__main__":
    main()