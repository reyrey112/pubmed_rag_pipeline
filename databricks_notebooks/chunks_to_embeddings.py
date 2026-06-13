from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd


def create_embeddings(chunks_table: str, embeddings_table: str):
    spark = SparkSession.builder.getOrCreate()

    # name of the model we want to use
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Create embeddings
    @pandas_udf(ArrayType(FloatType()))
    def embed_udf(texts: pd.Series) -> pd.Series:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(texts.tolist(), show_progress_bar=False)
        return pd.Series([e.tolist() for e in embeddings])

    df_chunks = spark.table(chunks_table)

    df_embedded = df_chunks.withColumn("embedding", embed_udf("chunk"))

    df_embedded.write.format("delta").mode("overwrite").saveAsTable(embeddings_table)

    print(f"Embedded {df_embedded.count()} chunks")

if __name__ == "__main__":
    create_embeddings("rag_pipeline.silver.chunks", "rag_pipeline.silver.embeddings")