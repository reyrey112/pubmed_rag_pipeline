from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, row_number, concat, lit, pandas_udf
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    abstract_table: str,
    chunks_table: str,
):
    spark = SparkSession.builder.getOrCreate()

    spark.sql("CREATE SCHEMA IF NOT EXISTS rag_pipeline.silver")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            pmid       STRING,
            chunk_id   STRING,
            chunk_index INT,
            chunk      STRING
        ) USING DELTA
    """)

    # Read raw abstracts from Delta Lake
    print("Read Abstracts from Table")
    df_abstracts = spark.table(f"{abstract_table}")

    # Only chunk abstracts for pmids that aren't already chunked, so
    # re-running this job doesn't reprocess the whole history each time.
    existing_pmids = spark.table(chunks_table).select("pmid").distinct()
    df_new_abstracts = df_abstracts.join(existing_pmids, on="pmid", how="left_anti")

    new_count = df_new_abstracts.count()
    if new_count == 0:
        print("No new abstracts to chunk. Skipping.")
        return

    print(f"Chunking {new_count} new abstracts (of {df_abstracts.count()} total)")

    # Define splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )

    @pandas_udf(ArrayType(StringType()))
    def chunk_udf(texts: pd.Series) -> pd.Series:
        return texts.apply(lambda t: splitter.split_text(t) if t else [])

    # Chunk and explode
    df_chunked = df_new_abstracts.withColumn(
        "chunks", chunk_udf("abstract")
    ).withColumn("chunk", explode("chunks"))

    # Add chunk metadata
    window = Window.partitionBy("pmid").orderBy("chunk")

    df_final = (
        df_chunked.withColumn("chunk_index", row_number().over(window) - 1)
        .withColumn("chunk_id", concat("pmid", lit("_chunk_"), "chunk_index"))
        .select("pmid", "chunk_id", "chunk_index", "chunk")
    )

    # Append new chunks to the silver layer. chunk_id is already unique per
    # pmid/chunk_index, and we've already excluded pmids that were chunked
    # before, so append (rather than overwrite) is safe and idempotent
    # with respect to already-processed abstracts.
    df_final.write.format("delta").mode("append").saveAsTable(chunks_table)
    print(f"Created {df_final.count()} chunks from {new_count} new papers")


if __name__ == "__main__":
    create_chunks("rag_pipeline.bronze.abstracts", "rag_pipeline.silver.chunks")