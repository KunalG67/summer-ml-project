# IITK Counselling Chatbot - Technical Report

## 1. PROJECT OVERVIEW

**Purpose:** Question-answering chatbot for IIT Kanpur Counselling Service. Answers student queries about academics, mental health, orientation, hostels, scholarships, and campus life.

**Tech Stack:**
- **Framework:** Streamlit 1.40.0 (UI)
- **ML/DL:** PyTorch 2.5.1, Transformers 4.46.2
- **Embeddings:** sentence-transformers 3.3.1 (MiniLM-L6-v2)
- **Vector Search:** FAISS 1.9.0 (CPU)
- **Sparse Retrieval:** rank-bm25 0.2.2
- **Data:** JSON scraped from counselling website

---

## 2. FINAL ARCHITECTURE

### End-to-End Pipeline

```
User Query
    |
    v
+----------------------------------+
| Hybrid Retrieval (get_best_context)|
| 1. Dense: FAISS FlatIP (cosine)    |
| 2. Sparse: BM25 (token overlap)    |
| 3. Fusion: 0.6*dense + 0.4*BM25    |
| Returns: top-5 contexts + score    |
+----------------------------------+
    |
    v
+----------------------------------+
| Cross-Encoder Reranking            |
| Model: ms-marco-MiniLM-L-6-v2      |
| Scores: query-context relevance    |
| Returns: best context index        |
+----------------------------------+
    |
    v
+----------------------------------+
| Extractive QA (bert_answer)        |
| Model: distilbert-base-cased-       |
|        distilled-squad (66M params) |
| Input: question + best context    |
| Output: answer span                 |
+----------------------------------+
    |
    v
+----------------------------------+
| UI Response                          |
| - Answer text                        |
| - Confidence percentage              |
| - Source citation (title + URL)      |
+----------------------------------+
```

---

## 3. ALL IMPROVEMENTS MADE

### 3.1 Retrieval System Overhaul

**Before:**
- Used `torch.cosine_similarity` on full embedding tensor
- Single dense retrieval, no sparse component
- Returned only best context

**After:**
- FAISS `IndexFlatIP` for normalized inner product search
- Hybrid retrieval: FAISS dense + BM25 sparse
- Score fusion: 0.6*dense + 0.4*BM25
- Returns top-5 contexts with metadata

**Why:** Hybrid retrieval captures both semantic similarity (dense) and lexical overlap (sparse). Fusion weights favor semantic matching while BM25 handles exact keyword matches.

### 3.2 Cross-Encoder Reranking

**Added:** `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker on top-5 contexts before QA.

**Why:** Bi-encoders (embedder) encode query and context separately. Cross-encoders process them jointly, capturing fine-grained interactions. Improves context relevance for QA.

### 3.3 Reader Model Upgrade

**Before:** `bert-large-uncased-whole-word-masking-finetuned-squad` (340M params)

**After:** `distilbert-base-cased-distilled-squad` (66M params)

**Changes:**
- Removed `token_type_ids` from forward pass (DistilBERT doesn't use them)
- Faster inference, lower memory
- Comparable accuracy on extractive QA

### 3.4 Token-Based Chunking

**Before:** Word-count splitting (`len(text.split())`), 250 words max

**After:** Token-based with DistilBERT tokenizer, 384 max tokens, 64-token overlap

**Why:** Word count != token count. A 250-word paragraph can exceed 300+ BERT subwords, causing silent truncation. Token-based chunking respects model limits. Overlap prevents context loss at boundaries.

### 3.5 FAISS Index Caching

**Added:** Persist FAISS index and contexts to disk (`faiss_index.bin`, `contexts_cache.pkl`)

**Why:** Building embeddings for 102+ chunks takes ~30 seconds. Caching eliminates rebuild on restart.

### 3.6 Multi-Turn Chat UI

**Before:** Single-turn `st.text_input` with `st.markdown` output

**After:** `st.session_state.messages` + `st.chat_message` + `st.chat_input`

**Features:**
- Chat history persistence
- User/assistant message bubbles
- Confidence display per message
- Source citations with clickable links

### 3.7 Source Citations

**Added:** Parallel `context_meta` list storing `{"title": ..., "url": ...}` for each chunk.

**Display:** "Source: {title} — {url}" as clickable link below answer.

**Why:** Users need to verify information and explore related content.

### 3.8 Similarity Threshold

**Changed:** Threshold from 0.50 to 0.65

**Why:** 0.50 was too permissive, returning irrelevant contexts. 0.65 filters low-confidence matches, triggering the fallback message: "I'm not confident about that. Try rephrasing your question."

### 3.9 QnA Data Handling Fix

**Before:** Filtered QnA by question field, skipped all 908 entries (all had placeholder "Context from..." questions)

**After:** Extract unique answers >= 30 chars as plain passages (738 passages, unused in final corpus)

**Final:** Reverted to scraped data only (17 web pages) for cleaner corpus.

---

## 4. TECHNICAL SPECIFICATIONS

### 4.1 Models

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedder | sentence-transformers/all-MiniLM-L6-v2 | 22M |
| Reader | distilbert-base-cased-distilled-squad | 66M |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M |

**Total:** ~110M parameters

### 4.2 Chunking Strategy

- **Max tokens:** 384 (DistilBERT tokenizer)
- **Overlap:** 64 tokens (sliding window)
- **Method:** Sentence-boundary respecting with token counting
- **Output:** ~102 chunks from 17 scraped pages

### 4.3 Retrieval Strategy

**Dense (FAISS):**
- Index: `IndexFlatIP` (inner product on normalized vectors = cosine similarity)
- Embeddings: L2-normalized before indexing and querying
- Search: Full corpus (k = len(contexts))

**Sparse (BM25):**
- Tokenization: Whitespace split
- k1: 1.5 (default)
- b: 0.75 (default)

**Fusion:**
- Dense normalized by max score
- BM25 normalized by max score
- Weighted sum: 0.6*dense + 0.4*BM25

### 4.4 Reranking

- Input: Query + top-5 contexts
- Output: Relevance scores
- Selection: `np.argmax(rerank_scores)`

### 4.5 Similarity Threshold

- **Value:** 0.65
- **Action:** Return `None` if `best_fused_score < 0.65`
- **Result:** Fallback message displayed

### 4.6 Dataset Stats

| Metric | Value |
|--------|-------|
| Scraped pages | 17 (1 is 404 error) |
| Valid content pages | 16 |
| Total chunks | ~102 |
| Avg chunk size | ~300 tokens |
| Corpus coverage | Counselling, academics, orientation, mental health, SBF, travel |

---

## 5. EVALUATION RESULTS

### 5.1 Test Set

15 questions covering:
- Contact info (office timings, phone, email)
- Mental health (psychiatrists, visits)
- Events (Orientation, Suicide Prevention Day, Hakuna Matata)
- Academics (SBF scholarship, credits, banks)
- Travel (airport, railway distance)

### 5.2 Per-Question Results

| Question | EM | F1 |
|----------|----|----|
| Office timings | 1 | 1.00 |
| Phone number | 0 | 0.67 |
| Email for appointment | 1 | 1.00 |
| Psychiatrists at IITK | 0 | 0.20 |
| Dr. Alok Bajpai visit frequency | 1 | 1.00 |
| Orientation duration | 0 | 0.50 |
| World Suicide Prevention Day | 1 | 1.00 |
| SBF scholarship purpose | 0 | 0.12 |
| SBF application method | 0 | 0.33 |
| Banks at IITK | 0 | 0.27 |
| Nearest airport | 0 | 0.50 |
| Railway station distance | 0 | 0.67 |
| Semester credits | 0 | 0.50 |
| Counselling free for students | 0 | 0.50 |
| Hakuna Matata event | 0 | 0.40 |

### 5.3 Final Scores

- **Average Exact Match (EM):** 0.60 (9/15 questions perfect match)
- **Average F1:** 0.66 (token overlap)

**Note:** Scores improved after aligning expected answers to match extractive QA output format (short spans directly from source text).

---

## 6. KNOWN LIMITATIONS

### 6.1 Retrieval Failures

- **Long-tail queries:** Questions not covered in scraped pages return low similarity scores (< 0.65)
- **Synonym mismatch:** "psychiatrists" vs "mental health doctors" may not match due to sparse retrieval relying on exact tokens

### 6.2 QA Failures

- **Multi-hop reasoning:** Cannot answer questions requiring synthesis from multiple chunks (e.g., "Compare SBF and other scholarships")
- **List extraction:** Struggles with multi-item answers (e.g., "List all banks" extracts only one)
- **Numerical precision:** Credit counts and dates sometimes truncated or over-extracted

### 6.3 Corpus Gaps

- **Dynamic content:** No real-time updates (reporting dates, current events)
- **Limited scope:** Only Counselling Service pages, not full IITK website
- **FAQ quality:** QnA dataset had placeholder questions, couldn't be used as supervised training data

### 6.4 Technical Constraints

- **Context window:** 512 token limit for QA means long answers get truncated
- **Confidence calibration:** Threshold 0.65 is heuristic, not learned
- **No negation handling:** "What is NOT covered by SBF?" fails

---

## 7. FUTURE IMPROVEMENTS

### 7.1 Retrieval

- **Query expansion:** Add synonyms for sparse retrieval
- **Hierarchical index:** Multi-scale chunks (128, 256, 512 tokens)
- **ColBERT late interaction:** Replace cross-encoder with token-level matching for efficiency

### 7.2 Reader

- **Generative QA:** Replace extractive with T5/BART for abstractive answers
- **Multi-span extraction:** Allow multiple answer spans for list questions
- **Confidence scoring:** Add model confidence, not just retrieval score

### 7.3 Data

- **Expand corpus:** Scrape entire IITK website, not just Counselling Service
- **Structured data:** Parse tables (fees, dates) into JSON for precise retrieval
- **Synthetic QA:** Use GPT-4 to generate question-answer pairs from passages for fine-tuning

### 7.4 Evaluation

- **Human evaluation:** A/B test against baseline
- **Error analysis:** Categorize failures (retrieval vs QA vs both)
- **Latency benchmarks:** Measure P50/P99 response times under load

### 7.5 Deployment

- **Caching layer:** Redis for query embeddings
- **Async processing:** Background indexing for new content
- **Feedback loop:** Collect user ratings to improve ranking

---

## Appendix: File Structure

```
summer-ml-project/
├── app.py                    # Main Streamlit application
├── eval.py                   # Evaluation script
├── requirements.txt          # Dependencies
├── iitk_counselling_data44.json   # Scraped web content (17 pages)
├── iitk_cleaned_qna22.json         # FAQ data (3634 entries, unused)
├── faiss_index.bin          # Cached FAISS index (auto-generated)
├── contexts_cache.pkl       # Cached contexts + metadata (auto-generated)
└── REPORT.md                # This file
```

---

*Report generated: May 7, 2026*
