from Bio import Entrez
import xml.etree.ElementTree as ET
import pandas as pd
import time, os
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.environ.get("EMAIL")


class PubSearch:
    def __init__(self) -> None:
        pass

    def search(self, query: str, max_results: int = 1000) -> list[str]:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        pmids = Entrez.read(handle)["IdList"]
        handle.close()
        return pmids

    # parse articles for information
    def parse(self, art: dict) -> dict:
        medline = art["MedlineCitation"]
        article = medline["Article"]

        # get authors
        authors = []
        for author in article.get("AuthorList", []):
            name = f"{author.get('LastName', '')} {author.get('ForeName', '')}".strip()
            authors.append(name)

        # get abstract
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

    # fetch articles in batches
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
        df = pd.DataFrame(articles)

        return df
