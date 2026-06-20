import argparse

from Bio import Entrez
import pandas as pd
import time, os

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

Entrez.email = dbutils.secrets.get(scope="rag_pipeline", key="EMAIL")


class PubSearch:
    def __init__(self) -> None:
        pass

    def search(self, query: str, max_results: int = 1000) -> list[str]:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        pmids = Entrez.read(handle)["IdList"]
        handle.close()
        return pmids

    def parse(self, art: dict) -> dict:
        medline = art["MedlineCitation"]
        article = medline["Article"]

        authors = []
        for author in article.get("AuthorList", []):
            name = f"{author.get('LastName', '')} {author.get('ForeName', '')}".strip()
            authors.append(name)

        abstract_text = ""
        if "Abstract" in article:
            abstract_text = " ".join(article["Abstract"].get("AbstractText", []))

        mesh_terms = [
            str(mesh["DescriptorName"]) for mesh in medline.get("MeshHeadingList", [])
        ]

        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", pub_date.get("MedlineDate", "Unknown"))

        return {
            "pmid": str(medline["PMID"]),
            "title": str(article.get("ArticleTitle", "")),
            "abstract": abstract_text,
            "authors": ", ".join(authors),
            "journal": str(art.get("Journal", {}).get("Title", "")),
            "year": str(year),
            "mesh_terms": ", ".join(mesh_terms),
            "doi": next(
                (
                    str(i)
                    for i in art.get("ELocationID", [])
                    if i.attributes.get("EIdType") == "doi"
                ),
                None,
            ),
        }

    def fetch(self, pmids: list[str], batch_size: int = 100) -> list[dict]:
        articles = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            ids = ",".join(batch)

            handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            for article in records["PubmedArticle"]:
                parsed_article = self.parse(article)
                articles.append(parsed_article)

            time.sleep(0.34)

        return articles

    def list_to_df(self, articles: list) -> pd.DataFrame:
        return pd.DataFrame(articles)


def write_to_delta_table(
    df: pd.DataFrame,
    meta_table: str,
    abstract_table: str,
):
    """Insert rows into a Databricks Delta table via Spark."""

    spark.sql("CREATE SCHEMA IF NOT EXISTS rag_pipeline.bronze")
    print("Schema rag_pipeline.bronze ready")

    meta_sdf = spark.createDataFrame(df.drop(columns=["abstract"]))
    abstract_sdf = spark.createDataFrame(df[["pmid", "abstract"]])

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {meta_table} (
            pmid       STRING,
            title      STRING,
            authors    STRING,
            journal    STRING,
            year       STRING,
            mesh_terms STRING,
            doi        STRING
        ) USING DELTA
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {abstract_table} (
            pmid     STRING,
            abstract STRING
        ) USING DELTA
    """)

    meta_sdf.write.format("delta").mode("append").saveAsTable(meta_table)
    abstract_sdf.write.format("delta").mode("append").saveAsTable(abstract_table)

    print(f"Inserted {len(df)} rows into {meta_table} and {abstract_table}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch PubMed articles and load into Databricks"
    )
    parser.add_argument("--query", default="Viral vectors", help="PubMed search query")
    parser.add_argument(
        "--max-results", type=int, default=500, help="Max number of articles to fetch"
    )
    parser.add_argument(
        "--meta_table",
        default="rag_pipeline.bronze.pubmed_meta",
        help="Target metadata table",
    )
    parser.add_argument(
        "--abstract_table",
        default="rag_pipeline.bronze.abstracts",
        help="Target abstracts table",
    )
    args = parser.parse_args()

    ps = PubSearch()

    print(f"Searching PubMed for: '{args.query}'")
    pmids = ps.search(args.query, args.max_results)
    print(f"Found {len(pmids)} articles")

    print("Fetching article details")
    articles = ps.fetch(pmids)

    print("Converting to DataFrame")
    df = ps.list_to_df(articles)

    print("Uploading to Databricks")
    write_to_delta_table(df, args.meta_table, args.abstract_table)

    print("Pipeline complete")


if __name__ == "__main__":
    main()
