from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd
import os


def create_embeddings(chunks_table: str, embeddings_table: str):
    spark = SparkSession.builder.getOrCreate()

    # Download model to shared volume (so all workers can access it)
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = (
        "/Volumes/rag_pipeline/silver/models/sentence_transformer_all-MiniLM-L6-v2"
    )

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

        config_file = os.path.join(MODEL_PATH, "config.json")

        if os.path.exists(config_file):
            print("Model Already Saved, Loading model")
        else:
            print("Downloading and Saving model")
            model = SentenceTransformer(MODEL_NAME)
            model.save(MODEL_PATH)
            print("Model saved")

        model = SentenceTransformer(MODEL_PATH)
        embeddings = model.encode(texts.tolist(), show_progress_bar=False)
        return pd.Series([e.tolist() for e in embeddings])

    df_chunks = spark.table(chunks_table)

    df_embedded = df_chunks.withColumn("embedding", embed_udf("chunk"))

    df_embedded.write.format("delta").mode("overwrite").saveAsTable(embeddings_table)
    spark.sql(
        f"ALTER TABLE {embeddings_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    print(f"Embedded {df_embedded.count()} chunks")


if __name__ == "__main__":
    create_embeddings("rag_pipeline.silver.chunks", "rag_pipeline.silver.embeddings")
