# IITK Counselling Chatbot

Extractive QA chatbot for IIT Kanpur Counselling Service using hybrid retrieval and cross-encoder reranking.

## What It Does

Answers student queries about:
- Counselling service (contact, appointments, psychiatrists)
- Academics (study techniques, SBF scholarship, credits)
- Mental health (anxiety, loneliness, procrastination)
- Orientation and campus events
- Travel and accommodation (airport, railway, hostels)

## Architecture

3-stage pipeline:

1. **Hybrid Retrieval** (`get_best_context`)
   - Dense: FAISS FlatIP with all-MiniLM-L6-v2 embeddings (cosine similarity)
   - Sparse: BM25Okapi on whitespace-tokenized contexts
   - Fusion: 0.6*dense + 0.4*BM25, returns top-5 contexts + score

2. **Cross-Encoder Reranking**
   - Model: ms-marco-MiniLM-L-6-v2
   - Scores query-context relevance, selects best context

3. **Extractive QA** (`bert_answer`)
   - Model: distilbert-base-cased-distilled-squad (66M params)
   - Input: question + reranked context (max 512 tokens)
   - Output: answer span extracted from context

**Features:**
- Multi-turn chat UI with session history
- Source citations (title + clickable URL)
- Confidence threshold: 0.65 (falls back to "not confident" message)
- Token-based chunking: 384 max tokens, 64-token overlap
- FAISS index caching to disk for fast restart

## Models

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedder | sentence-transformers/all-MiniLM-L6-v2 | 22M |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M |
| Reader | distilbert-base-cased-distilled-squad | 66M |

Total: ~110M parameters

## Evaluation

Test set: 15 questions (contact info, mental health, events, academics, travel)

- **Exact Match (EM):** 0.60
- **F1 Score:** 0.66

Scores measured with token-level F1 comparing predicted answer spans against extractive targets from source text.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requirements: Python 3.10+, 4GB RAM, internet for model download on first run.

## File Structure

```
summer-ml-project/
├── app.py                      # Main Streamlit app
├── eval.py                     # Evaluation script
├── requirements.txt            # Dependencies
├── iitk_counselling_data44.json    # Scraped content (16 valid pages, ~102 chunks)
├── iitk_cleaned_qna22.json          # FAQ data (unused in current corpus)
├── REPORT.md                   # Full technical report
├── faiss_index.bin            # Cached FAISS index (auto-generated)
└── contexts_cache.pkl         # Cached contexts + metadata (auto-generated)
```

Data source: Scraped from https://www.iitk.ac.in/counsel/
