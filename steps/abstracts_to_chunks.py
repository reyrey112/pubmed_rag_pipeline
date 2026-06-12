from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, row_number, concat, lit, pandas_udf
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from delta import configure_spark_with_delta_pip

load_dotenv()


def create_chunks(
    abstract_table: str,
    chunks_table: str,
):
    print("Start Spark")
    spark = DatabricksSession.builder.remote(
        host=f"{os.environ.get("DATABRICKS_HOST")}",
        access_token=f"{os.environ.get("DATABRICKS_TOKEN")}",
    ).getOrCreate()
    # Read raw abstracts from Delta Lake
    print("Read Abstracts from Table")
    df_abstracts = spark.table("rag_pipeline.bronze.abstracts")
    # Define splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )

    @pandas_udf(ArrayType(StringType()))
    def chunk_udf(texts: pd.Series) -> pd.Series:
        return texts.apply(lambda t: splitter.split_text(t) if t else [])

    # Chunk and explode
    df_chunked = df_abstracts.withColumn(
        "chunks", chunk_udf("abstract_text")
    ).withColumn("chunk", explode("chunks"))

    # Add chunk metadata
    window = Window.partitionBy("pmid").orderBy("chunk")

    df_final = (
        df_chunked.withColumn("chunk_index", row_number().over(window) - 1)
        .withColumn("chunk_id", concat("pmid", lit("_chunk_"), "chunk_index"))
        .select("pmid", "chunk_id", "chunk_index", "chunk")
    )

    # Write to silver layer in Delta Lake
    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS {os.environ.get("DATABRICKS_CATALOG")}.silver"
    )
    df_final.write.format("delta").mode("overwrite").saveAsTable(chunks_table)
    print(f"Created {df_final.count()} chunks from {df_abstracts.count()} papers")
