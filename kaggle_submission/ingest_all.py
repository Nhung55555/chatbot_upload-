# ingest_all.py
from main1 import ingest, save_faiss, Embedder
from ingest_pubmed_diabetes import ingest_pubmed
from pathlib import Path

def ingest_all():
    # 1) PDF
    print("Ingesting PDFs...")
    ingest("pdfs/")   # thư mục PDF của bạn

    # 2) PubMed
    print("Ingesting PubMed abstracts...")
    pubmed_docs = ingest_pubmed()

    # 3) Load lại PDF chunks
    from main1 import process_pdf
    pdf_docs = []
    for pdf in Path("pdfs").glob("*.pdf"):
        pdf_docs.extend(process_pdf(str(pdf)))

    all_docs = pdf_docs + pubmed_docs
    print(f"TOTAL DOCS = {len(all_docs)}")

    embedder = Embedder("sentence-transformers/all-mpnet-base-v2")
    save_faiss(all_docs, embedder)

if __name__ == "__main__":
    ingest_all()
