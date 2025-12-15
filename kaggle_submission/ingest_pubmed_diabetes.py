# ingest_pubmed_diabetes.py
from Bio import Entrez
from langchain.schema import Document
from tqdm import tqdm
import time

Entrez.email = "ptcnhung99tn@gmail.com"   # BẮT BUỘC

QUERY = (
    '"Diabetes Mellitus"[MeSH Terms] '
    'OR "Type 2 Diabetes Mellitus"[MeSH Terms] '
    'OR "Type 1 Diabetes Mellitus"[MeSH Terms]'
)

MAX_RESULTS = 5000   # BẮT ĐẦU NHỎ

def fetch_pubmed_ids():
    handle = Entrez.esearch(
        db="pubmed",
        term=QUERY,
        retmax=MAX_RESULTS
    )
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_abstract(pmid):
    handle = Entrez.efetch(
        db="pubmed",
        id=pmid,
        rettype="abstract",
        retmode="xml"
    )
    record = Entrez.read(handle)
    return record

def parse_abstract(record):
    articles = record["PubmedArticle"]
    docs = []

    for art in articles:
        try:
            article = art["MedlineCitation"]["Article"]
            title = article.get("ArticleTitle", "")
            abstract = article["Abstract"]["AbstractText"]
            text = title + "\n" + " ".join(abstract)

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "PubMed",
                    "source_file": "PubMed_Abstract",
                    "pmid": art["MedlineCitation"]["PMID"],
                    "document_type": "abstract",
                    "medical_domain": "endocrinology"
                }
            ))
        except:
            continue

    return docs

def ingest_pubmed():
    pmids = fetch_pubmed_ids()
    print(f"Fetched {len(pmids)} PubMed IDs")

    all_docs = []

    for pmid in tqdm(pmids):
        try:
            record = fetch_abstract(pmid)
            docs = parse_abstract(record)
            all_docs.extend(docs)
            time.sleep(0.35)  # rate limit
        except:
            continue

    return all_docs

if __name__ == "__main__":
    docs = ingest_pubmed()
    print(f"Total abstracts: {len(docs)}")
