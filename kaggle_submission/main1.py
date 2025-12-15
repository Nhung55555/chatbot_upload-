import os
import json
import fitz
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from langchain.schema import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

import re
import torch
from kaggle_secrets import UserSecretsClient
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

DEBUG = True   # Bật tắt debug ở đây

# Load GROQ_API_KEY from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    groq_api_key = user_secrets.get_secret("GROQ_API_KEY") 
    os.environ["GROQ_API_KEY"] = groq_api_key
except Exception:
    # Xử lý nếu không tìm thấy key trong Kaggle Secrets (ví dụ: đang chạy local)
    pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# ======================================
# CONFIG
# ======================================
CONFIG = {
    "chunk_size": 1800,
    "chunk_overlap": 300,

    "embed_model": "sentence-transformers/all-mpnet-base-v2",
    "reranker_model": "ncbi/MedCPT-Cross-Encoder",
    "nli_model": "typeform/distilbert-base-uncased-mnli",

    "fetch_k": 60,
    "rerank_k": 15,
    "final_k": 8,

    "n_workers": 4,
    "shard_dir": "vectorstore/shards",
}

# ======================================
# UTIL
# ======================================
def clean_text(s: str):

    if not s:
        return ""
    return " ".join(s.replace("\u200b", "").replace("\x0c", " ").split())

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def log_retrieval(query, docs):
    ensure_dir("logs")
    with open("logs/retrieval_log.jsonl", "a", encoding="utf-8") as f:
        record = {
            "query": query,
            "retrieved_docs": [
                {
                    "file": d.metadata.get("source_file"),
                    "page_start": d.metadata.get("page_start"),
                    "page_end": d.metadata.get("page_end")
                }
                for d in docs
            ]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        
# chuẩn hóa thuật ngữ y khoa trong truy vấn 
def normalize_medical_query(q: str):
    q = q.lower().strip()

    medical_map = {
        # Diabetes terminology
        "diabetes": "diabetes mellitus",
        "type 1 diabetes": "type 1 diabetes mellitus",
        "type 2 diabetes": "type 2 diabetes mellitus",
        "sugar disease": "diabetes mellitus",

        # Diagnosis
        "how to diagnose": "diagnostic criteria for",
        "diagnosis of": "diagnostic criteria for",
        "how is diagnosed": "diagnostic criteria for",
        "test for": "diagnostic tests for",
        "screening for": "screening guidelines for",

        # Treatment
        "how to treat": "treatment guidelines for",
        "treatment of": "treatment guidelines for",
        "therapy for": "treatment guidelines for",
        "drug for": "pharmacological treatment for",
        "medication for": "pharmacological treatment for",
        "insulin use": "insulin therapy guidelines",

        # Lab measurements
        "blood sugar": "blood glucose",
        "fasting sugar": "fasting plasma glucose",
        "postprandial sugar": "postprandial blood glucose",
        "hba1c": "a1c", # <-- CẢI TIẾN: Ánh xạ HbA1c sang A1C (dùng thuật ngữ chính trong ADA)
        "a1c": "glycated hemoglobin", # # <-- CẢI TIẾN: Đảm bảo A1C cũng được mở rộng nếu cần
        # "hba1c": "glycated hemoglobin (HbA1c)",
        "glycated hemoglobin": "glycated hemoglobin",# <-- CẢI TIẾN:

        # Complications
        "eye damage": "diabetic retinopathy",
        "kidney damage": "diabetic nephropathy",
        "nerve damage": "diabetic neuropathy",
        "heart disease": "cardiovascular disease in diabetes",

        # Risk & prevention
        "prevent": "prevention strategies for",
        "risk of": "risk factors for",
        "cause of": "etiology of"
    }

    # for k, v in medical_map.items():
    #     if k in q:
    #         q = q.replace(k, v)

    # return q

    # Phải thực hiện mapping trên các từ khóa gốc trước
    # Logic cũ của bạn cần được tối ưu hóa một chút để tránh trùng lặp
    for k, v in medical_map.items():
        if k in q:
            q = q.replace(k, v)

    # Chạy lại normalization sau khi mapping (ví dụ: a1c -> glycated hemoglobin)
    for k, v in medical_map.items():
        if k in q:
            q = q.replace(k, v)

    return q

# ======================================
# PDF PROCESSING
# ======================================
def extract_page_text(pdf, p):
    try:
        page = pdf.load_page(p)
        return page.get_text("text") or ""
    except:
        return ""

# def semantic_chunk(text, max_len=2000, overlap=300):
#     paras = re.split(r"\n{2,}", text)
#     chunks = []
#     buff = ""

#     for p in paras:
#         if len(buff) + len(p) < max_len:
#             buff += " " + p
#         else:
#             chunks.append(buff.strip())
#             buff = buff[-overlap:] + " " + p

#     if buff.strip():
#         chunks.append(buff.strip())

#     return chunks


# def chunk_with_metadata(pages, pdf_path):
#     combined_text = []
#     page_map = []

#     for p in pages:
#         clean = clean_text(p["text"])
#         combined_text.append(clean)
#         page_map.append(p["page"] + 1)

#     full_text = "\n\n".join(combined_text)
#     chunks = semantic_chunk(
#         full_text,
#         max_len=CONFIG["chunk_size"],
#         overlap=CONFIG["chunk_overlap"]
#     )

#     docs = []
#     for chunk in chunks:
#         docs.append(
#             Document(
#                 page_content=chunk,
#                 metadata={
#                     "source": pdf_path,
#                     "source_file": os.path.basename(pdf_path),
#                     "page_start": page_map[0],
#                     "page_end": page_map[-1],
#                     "document_type": "diabetes_guideline",
#                     "medical_domain": "endocrinology",
#                     "year": 2025
#                 }
#             )
#         )
#     return docs


# KHÔNG CẦN CHỈNH SỬA LOGIC BÊN TRONG, CHỈ THAY ĐỔI ĐẦU VÀO HÀM
def semantic_chunk(text, max_len, overlap):
    # Logic của bạn
    paras = re.split(r"\n{2,}", text)
    chunks = []
    buff = ""

    for p in paras:
        if len(buff) + len(p) < max_len:
            buff += " " + p
        else:
            chunks.append(buff.strip())
            buff = buff[-overlap:] + " " + p

    if buff.strip():
        chunks.append(buff.strip())

    return chunks

# HÀM NÀY CẦN CHỈNH SỬA ĐỂ NHẬN THAM SỐ MỚI
def chunk_with_metadata(pages, pdf_path, chunk_size, chunk_overlap): # <-- ĐÃ THÊM THAM SỐ
    combined_text = []
    page_map = []
    
    for p in pages:
        clean = clean_text(p["text"])
        combined_text.append(clean)
        page_map.append(p["page"] + 1)
        
    full_text = "\n\n".join(combined_text)
    
    # SỬ DỤNG THAM SỐ ĐẦU VÀO
    chunks = semantic_chunk(
        full_text,
        max_len=chunk_size, 
        overlap=chunk_overlap 
    )

    docs = []
    # ... (giữ nguyên logic tạo Document) ...
    for chunk in chunks:
        # Xác định phạm vi trang của chunk này (có thể cần logic phức tạp hơn, nhưng giữ đơn giản cho đồ án)
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path,
                    "source_file": os.path.basename(pdf_path),
                    "page_start": page_map[0],
                    "page_end": page_map[-1],
                    "document_type": "diabetes_guideline",
                    "medical_domain": "endocrinology",
                    "year": 2025
                }
            )
        )
    return docs


# def process_pdf(pdf_path):
#     pdf = fitz.open(pdf_path)
#     n_pages = pdf.page_count

#     all_chunks = []
#     block_size = 40

#     for start in tqdm(range(0, n_pages, block_size), desc=f"Reading {pdf_path}"):
#         end = min(n_pages - 1, start + block_size - 1)
#         pages = []
#         for p in range(start, end + 1):
#             pages.append({
#                 "page": p,
#                 "text": extract_page_text(pdf, p)
#             })
#         chunks = chunk_with_metadata(pages, pdf_path)
#         all_chunks.extend(chunks)

#     return all_chunks

# HÀM NÀY CẦN CHỈNH SỬA
def process_pdf(pdf_path):
    pdf = fitz.open(pdf_path)
    n_pages = pdf.page_count
    file_name = os.path.basename(pdf_path)

    # ==================================================
    # CẢI TIẾN 1: LOGIC MULTI-STRATEGY CHUNKING
    # ==================================================
    if "ADA" in file_name or "Standards-of-Care" in file_name:
        # CPG: Kích thước nhỏ hơn để bắt thông tin số học nằm gần Bảng/Hình
        current_chunk_size = 500
        current_chunk_overlap = 100
        print(f"Applying SMALL Chunking ({current_chunk_size}/{current_chunk_overlap}) for CPG: {file_name}")
    else:
        # Textbook: Kích thước vừa phải cho kiến thức nền tảng (giảm nhẹ so với 1800)
        current_chunk_size = 1200
        current_chunk_overlap = 300
        print(f"Applying MEDIUM Chunking ({current_chunk_size}/{current_chunk_overlap}) for Textbook: {file_name}")
    # ==================================================

    all_chunks = []
    block_size = 40 # Giữ nguyên logic xử lý từng khối trang

    for start in tqdm(range(0, n_pages, block_size), desc=f"Reading {pdf_path}"):
        end = min(n_pages - 1, start + block_size - 1)
        pages = []
        for p in range(start, end + 1):
            pages.append({
                "page": p,
                "text": extract_page_text(pdf, p)
            })
        
        # TRUYỀN THAM SỐ MỚI VÀO HÀM CHUNKING
        chunks = chunk_with_metadata(
            pages, 
            pdf_path,
            current_chunk_size, 
            current_chunk_overlap
        )
        all_chunks.extend(chunks)

    return all_chunks

# ======================================
# EMBEDDING + FAISS
# ======================================
class Embedder:
    def __init__(self, model):
        self.model = SentenceTransformer(model, device=DEVICE)

    def encode(self, texts):
        embs = self.model.encode(texts, batch_size=64, convert_to_numpy=True)
        return embs.astype("float32")

# ======================================
# Sửa Hàm save_faiss
# ======================================

def save_faiss(docs, embedder):
    ensure_dir(CONFIG["shard_dir"])
    texts = [d.page_content for d in docs]
    embs = embedder.encode(texts)

    metas = [d.metadata.copy() for d in docs]
    for i, meta in enumerate(metas):
        meta["content"] = texts[i]
        meta["__id"] = int(i)

    dim = embs.shape[1]
    flat = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIDMap(flat)
    ids = np.arange(len(embs)).astype('int64')
    idx.add_with_ids(embs, ids)

    faiss.write_index(idx, f"{CONFIG['shard_dir']}/index.index")
    with open(f"{CONFIG['shard_dir']}/meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def simple_tokenize(s):
    s = re.sub(r'[^0-9a-zA-Z\u00C0-\u017F\s]', ' ', s)  # keep unicode letters
    return s.lower().split()


def load_faiss():
    idx = faiss.read_index(f"{CONFIG['shard_dir']}/index.index")
    metas = {}
    corpus = []

    with open(f"{CONFIG['shard_dir']}/meta.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            metas[int(m["__id"])] = m
            # corpus.append(m.get("content", "").lower().split())
            corpus.append(simple_tokenize(m.get("content","")))

    bm25 = BM25Okapi(corpus)
    return idx, metas, bm25



# ======================================
# RERANKER + NLI + LLM
# ======================================
class Reranker:
    def __init__(self):
        device_idx = 0 if DEVICE == "cuda" else -1
        self.model = CrossEncoder(CONFIG["reranker_model"],
                                  device = device_idx)

    def rerank(self, query, docs):
        pairs = [[query, d.page_content[:1500]] for d in docs]
        scores = self.model.predict(pairs, batch_size = 8)
        order = np.argsort(scores)[::-1]
        return [docs[i] for i in order[:CONFIG["rerank_k"]]]

class Verifier:
    def __init__(self):
        device_idx = 0 if DEVICE == "cuda" else -1
        self.pipe = pipeline(
                    "text-classification",
                    model=CONFIG["nli_model"],
                    device=device_idx,
                    top_k=None
                )
                            #  return_all_scores=True)
    def entailment(self, premise, hypothesis):
        out = self.pipe(premise + " </s> " + hypothesis)

        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
            outs = out[0]

        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            outs = out
            
        else: 
            outs = [out]
            
        for o in outs:
            if "entail" in o["label"].lower():
                return float(o.get("score", 0.0))
        return 0.0

def call_llm(prompt):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.05,
        max_tokens=1024
    )
    resp = llm.invoke(prompt)
    return resp.content


# ======================================
# BUILD ANSWER
# ======================================
def build_prompt(query, docs):
    prompt = (
        "You are a strict medical assistant. Absolutely no outside knowledge is used.\n"
        "Answer only based on the information contained in the <CONTEXT> section below.\n"
        "For questions about quantity or simple inference, counting/inferring from [CONTEXT] is required.\n"
        "If information is absolutely absent, answer precisely: 'There is no evidence in the material provided.'\n"
        "All factual statements MUST include citation: (Source: <file>, pages <x>-<y>)\n\n"
        f"Question: {query}\n\nCONTEXT:\n"
    )
    for i,d in enumerate(docs,1):
        prompt += f"\n[DOC {i}] {d.metadata['source_file']} (pages {d.metadata['page_start']}-{d.metadata['page_end']})\n"
        prompt += d.page_content[:2000] + "\n"
    prompt += "\nAnswer:"
    return prompt


# ======================================
# CHAT LOOP
# ======================================
def chat():
    print("Clinical RAG (VSCode) Ready. Type 'exit' to quit.\n")

    # idx, metas = load_faiss()
    idx, metas, bm25 = load_faiss()
    embedder = Embedder(CONFIG["embed_model"])
    reranker = Reranker()
    verifier = Verifier()

    while True:
        # q = input("You: ").strip()
        raw_q = input("You: ").strip()
        q = normalize_medical_query(raw_q)

        if q.lower() == "exit":
            break

        if DEBUG:
            print("\n====================")
            print(" STEP 1 — QUERY NORMALIZATION")
            print("====================")
            print("Raw query:", raw_q)
            print("Normalized query:", q)

        # 1) Embed query & search
        q_emb = embedder.encode([q])
        # D, I = idx.search(q_emb, CONFIG["fetch_k"])
        # Dense retrieval
        D_dense, I_dense = idx.search(q_emb, CONFIG["fetch_k"])

        if DEBUG:
            print("\n====================")
            print(" STEP 2 — DENSE RETRIEVAL (FAISS)")
            print("====================")
            for rank, (dist, idx_) in enumerate(zip(D_dense[0][:10], I_dense[0][:10])):
                print(f"Rank {rank+1}: ID={idx_}, dist={dist}")


        # Keyword retrieval (BM25)
        tokenized_q = q.lower().split()
        bm25_scores = bm25.get_scores(tokenized_q)
        I_bm25 = np.argsort(bm25_scores)[::-1][:CONFIG["fetch_k"]]


        if DEBUG:
            print("\n====================")
            print(" STEP 3 — BM25 RETRIEVAL")
            print("====================")
            for rank, idx_ in enumerate(I_bm25[:10]):
                print(f"Rank {rank+1}: ID={idx_}, score={bm25_scores[idx_]}")



        # Hợp nhất chỉ mục
        # I = list(set(I_dense[0].tolist() + I_bm25.tolist()))
        seen = set()
        I_ordered = []
        for idx_ in list(I_dense[0]) + list(I_bm25):
            if idx_ not in seen:
                seen.add(idx_)
                I_ordered.append(idx_)
                
        if DEBUG:
            print("\n====================")
            print(" STEP 4 — MERGED CANDIDATES (ORDERED)")
            print("====================")
            print(I_ordered[:20])
        

        candidates = []
        for j in I_ordered:
            meta = metas[j]
           # Lấy nội dung chunk đã được làm sạch và lưu trong metadata
            chunk_content = meta.get("content", "") 
            
            if chunk_content: # Nếu tìm thấy nội dung
                candidates.append(Document(page_content=chunk_content, metadata=meta))
        
        # 2) Rerank
        top_docs = reranker.rerank(q, candidates)

        if DEBUG:
            print("\n====================")
            print(" STEP 5 — RERANKER SCORES")
            print("====================")

            pairs = [[q, d.page_content[:1500]] for d in candidates]
            scores = reranker.model.predict(pairs, batch_size=8)

            # Print top 10 sorted by score
            order = np.argsort(scores)[::-1]
            for i in order[:10]:
                m = candidates[i].metadata
                print(f"Score={scores[i]:.4f} | File={m['source_file']} pages {m['page_start']}-{m['page_end']}")


        
        # final_docs = top_docs[:CONFIG["final_k"]]
        final_docs = []
        used_sources = {}

        for d in top_docs:
            src = d.metadata["source_file"]
            used_sources.setdefault(src, 0)

            if used_sources[src] < 2:
                final_docs.append(d)
                used_sources[src] += 1

            if len(final_docs) >= CONFIG["final_k"]:
                break


        if DEBUG:
            print("\n====================")
            print(" STEP 6 — FINAL DOCS FOR LLM")
            print("====================")
            for i, d in enumerate(final_docs, 1):
                m = d.metadata
                print(f"[DOC {i}] {m['source_file']} pages {m['page_start']}-{m['page_end']}")
                print("Content preview:", d.page_content[:200], "...\n")

        # LOG PHỤC VỤ ĐÁNH GIÁ LUẬN VĂN
        # log_retrieval(q, final_docs)


        # 3) Prompt → LLM
        prompt = build_prompt(q, final_docs)
        
        if DEBUG:
            print("\n====================")
            print(" STEP 7 — PROMPT TO LLM")
            print("====================")
            print(prompt[:1500], "...\n")
            
        raw_answer = call_llm(prompt)

        if DEBUG:
            print("\n====================")
            print("STEP 8 — RAW LLM ANSWER")
            print("====================")
            print(raw_answer)

        # 4) Verify each sentence
        # sentences = [s.strip() for s in raw_answer.split(".") if len(s.strip()) > 10]
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_answer) if len(s.strip()) > 10]
        unsupported = []

        if DEBUG:
            print("\n====================")
            print(" STEP 9 — NLI VERIFICATION")
            print("====================")


        for s in sentences:
            ok = False
            for d in final_docs:
                score = verifier.entailment(d.page_content[:1200], s)

                if DEBUG:
                    print(f"\nHypothesis: {s}")
                    print(f"Premise: {d.page_content[:200].rstrip()}...")
                    print("NLI score:", score)
                if score > 0.6:
                    ok = True
                    break
            if not ok:
                unsupported.append(s)

        # 5) Output
        print("\n--- ANSWER ---\n")
        print(raw_answer)

        print("\n--- CITATIONS ---")
        cited = set()
        for d in final_docs:
            key = f"{d.metadata['source_file']}-{d.metadata['page_start']}-{d.metadata['page_end']}"
            if key not in cited:
                print(f"- {d.metadata['source_file']} (pages {d.metadata['page_start']}-{d.metadata['page_end']})")
                cited.add(key)

        # print("\n--- VERIFICATION ---")
        # if unsupported:
        #     print("Unsupported claims:")
        #     for uc in unsupported:
        #         print(" -", uc)
        # else:
        #     print("All claims supported by documents.")
        print("\n--- VERIFICATION ---")
        if unsupported:
            total_claims = len(sentences) # sentences là tổng số câu khẳng định
            unsupported_count = len(unsupported)
            supported_count = total_claims - unsupported_count
            
            # TÍNH TOÁN FACTUALITY SCORE Ở ĐÂY
            factuality_score = supported_count / total_claims if total_claims > 0 else 0.0

            print(f"Factuality Score: {factuality_score:.3f} ({supported_count}/{total_claims} claims supported)")
            print("Unsupported claims:")
            for uc in unsupported:
                print(" -", uc)
        else:
            print("All claims supported by documents.")
            print("Factuality Score: 1.000 (All claims supported)")


# ======================================
# INGEST
# ======================================
def ingest(pdf_dir):
    all_docs = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))

    for pdf in pdf_files:
        print(f"\nExtracting: {pdf}")
        chunks = process_pdf(str(pdf))
        all_docs.extend(chunks)

    # Embed + save FAISS
    embedder = Embedder(CONFIG["embed_model"])
    print("Embedding & saving FAISS...")
    save_faiss(all_docs, embedder)
    print("Done.")


# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    load_dotenv() 
    print("\nOPTIONS:")
    print("1) Ingest PDFs: python clinical_rag_vscode.py ingest <your_folder>")
    print("2) Chat:        python clinical_rag_vscode.py chat\n")

    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "ingest":
        ingest(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "chat":
        chat()
    else:
        print("Invalid command.")
