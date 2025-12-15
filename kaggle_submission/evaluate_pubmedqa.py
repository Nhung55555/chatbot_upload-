# evaluate_pubmedqa.py
from langchain.schema import Document
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import re
import json

# IMPORT pipeline của bạn
from main1 import (
    CONFIG,
    load_faiss,
    Embedder,
    Reranker,
    Verifier,
    build_prompt,
    call_llm,
    normalize_medical_query
)

def load_diabetes_pubmedqa():
    dataset = load_dataset("pubmed_qa", "pqa_labeled")["train"]

    keywords = [
        "diabetes", "diabetes mellitus",
        "type 1 diabetes", "type 2 diabetes",
        "hba1c", "blood glucose", "insulin"
    ]

    def is_diabetes(ex):
        text = (ex["question"] + " " + ex["context"]).lower()
        return any(k in text for k in keywords)

    data = [ex for ex in dataset if is_diabetes(ex)]
    print(f"Loaded {len(data)} diabetes PubMedQA questions")
    return data


def run_rag(question, idx, metas, bm25, embedder, reranker, verifier):
    q = normalize_medical_query(question)

    q_emb = embedder.encode([q])
    _, I_dense = idx.search(q_emb, CONFIG["fetch_k"])

    bm25_scores = bm25.get_scores(q.lower().split())
    I_bm25 = np.argsort(bm25_scores)[::-1][:CONFIG["fetch_k"]]

    seen, merged = set(), []
    for i in list(I_dense[0]) + list(I_bm25):
        if i not in seen:
            seen.add(i)
            merged.append(i)

    candidates = []
    for i in merged:
        meta = metas[i]
        if meta.get("content"):
            candidates.append(Document(
                page_content=meta["content"],
                metadata=meta
            ))

    docs = reranker.rerank(q, candidates)[:CONFIG["final_k"]]

    prompt = build_prompt(q, docs)
    answer = call_llm(prompt)

    return answer


def normalize_decision(text):
    text = text.lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "maybe"


def evaluate():
    data = load_diabetes_pubmedqa()

    idx, metas, bm25 = load_faiss()
    embedder = Embedder(CONFIG["embed_model"])
    reranker = Reranker()
    verifier = Verifier()

    correct = 0

    for ex in tqdm(data):
        pred = run_rag(
            ex["question"],
            idx, metas, bm25,
            embedder, reranker, verifier
        )

        if normalize_decision(pred) == ex["final_decision"].lower():
            correct += 1

    print(f"Accuracy: {correct}/{len(data)} = {correct/len(data):.3f}")

