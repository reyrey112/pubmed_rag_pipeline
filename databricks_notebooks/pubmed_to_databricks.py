import argparse

from Bio import Entrez
import pandas as pd
import time, os

from pyspark.sql import SparkSession
from delta.tables import DeltaTable

spark = SparkSession.builder.getOrCreate()

Entrez.email = dbutils.secrets.get(scope="rag_pipeline", key="EMAIL")

PMID_REGISTRY_TABLE = "rag_pipeline.bronze.pmid_registry"


class PubSearch:
    def __init__(self) -> None:
        pass

    def search(
        self,
        query: str,
        max_results: int = 5000,
        mindate: str | None = None,
        maxdate: str | None = None,
    ) -> list[str]:
        """
        If mindate/maxdate are given (YYYY/MM/DD or YYYY-MM-DD), the search
        is bounded to that window using Entrez date (edat) -- i.e. when
        PubMed indexed the record, not when it was published. This ensures
        successive runs of the same term pull newly-indexed articles
        instead of re-fetching the same top-N results every time.
        """
        search_kwargs = {"db": "pubmed", "term": query, "retmax": max_results}

        if mindate or maxdate:
            search_kwargs["datetype"] = "edat"
            search_kwargs["mindate"] = mindate or "1900/01/01"
            search_kwargs["maxdate"] = maxdate or time.strftime("%Y/%m/%d")

        handle = Entrez.esearch(**search_kwargs)
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


def ensure_registry_table():
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {PMID_REGISTRY_TABLE} (
            pmid STRING
        ) USING DELTA
    """)


def filter_new_pmids(pmids: list[str]) -> list[str]:
    """
    Anti-join the searched pmids against the (skinny) pmid_registry table
    in Spark, rather than collecting the registry to the driver. Scales
    with the cluster instead of driver memory as the registry grows.
    """
    ensure_registry_table()

    if not pmids:
        return []

    searched_df = spark.createDataFrame([(p,) for p in pmids], ["pmid"])
    registry_df = spark.table(PMID_REGISTRY_TABLE).select("pmid")

    new_pmids_df = searched_df.join(registry_df, on="pmid", how="left_anti")
    new_pmids = [row.pmid for row in new_pmids_df.collect()]

    print(f"{len(pmids)} pmids searched, {len(new_pmids)} are new (not in registry)")
    return new_pmids


def write_to_delta_table(
    df: pd.DataFrame,
    meta_table: str,
    abstract_table: str,
):
    """Merge rows into Databricks Delta tables via Spark, keyed on pmid.

    Using MERGE (instead of append) makes ingestion idempotent: even if a
    pmid slips through the registry anti-join (e.g. a race between
    concurrent runs), it will not create a duplicate row.
    """

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

    meta_delta = DeltaTable.forName(spark, meta_table)
    meta_delta.alias("t").merge(
        meta_sdf.alias("s"), "t.pmid = s.pmid"
    ).whenNotMatchedInsertAll().execute()

    abstract_delta = DeltaTable.forName(spark, abstract_table)
    abstract_delta.alias("t").merge(
        abstract_sdf.alias("s"), "t.pmid = s.pmid"
    ).whenNotMatchedInsertAll().execute()

    print(f"Merged {len(df)} rows into {meta_table} and {abstract_table}")

    # Update the registry so future runs (any search term) skip these pmids
    ensure_registry_table()
    registry_delta = DeltaTable.forName(spark, PMID_REGISTRY_TABLE)
    new_pmid_sdf = df[["pmid"]].pipe(lambda d: spark.createDataFrame(d))
    registry_delta.alias("t").merge(
        new_pmid_sdf.alias("s"), "t.pmid = s.pmid"
    ).whenNotMatchedInsertAll().execute()

    print(f"Registered {len(df)} pmids in {PMID_REGISTRY_TABLE}")


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
    parser.add_argument(
        "--mindate",
        default=None,
        help="Only fetch articles Entrez-indexed on/after this date (YYYY/MM/DD)",
    )
    parser.add_argument(
        "--maxdate",
        default=None,
        help="Only fetch articles Entrez-indexed on/before this date (YYYY/MM/DD)",
    )
    args = parser.parse_args()

    ps = PubSearch()

    print(f"Searching PubMed for: '{args.query}' (mindate={args.mindate}, maxdate={args.maxdate})")
    pmids = ps.search(args.query, args.max_results, mindate=args.mindate, maxdate=args.maxdate)
    print(f"Found {len(pmids)} articles")

    print("Filtering out pmids already in the registry")
    new_pmids = filter_new_pmids(pmids)

    if not new_pmids:
        print("No new articles to fetch. Pipeline complete")
        return

    print("Fetching article details")
    articles = ps.fetch(new_pmids)

    print("Converting to DataFrame")
    df = ps.list_to_df(articles)

    print("Uploading to Databricks")
    write_to_delta_table(df, args.meta_table, args.abstract_table)

    print("Pipeline complete")


if __name__ == "__main__":
    main()