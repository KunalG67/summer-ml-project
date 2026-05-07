

import streamlit as st
import json
import pickle
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util, CrossEncoder


try:
    with open("iitk_cleaned_qna22.json", "r", encoding="utf-8") as f2:
        qna = json.load(f2)

    for entry in qna:
        for key in ("answer",):
            if key in entry and isinstance(entry[key], str):
                try:
                    entry[key] = entry[key].encode("latin-1").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass

    seen = set()
    qna_passages = []
    for pair in qna:
        a = pair.get("answer", "")
        if len(a) >= 30 and a not in seen:
            seen.add(a)
            qna_passages.append(a)
    qna = qna_passages

    with open("iitk_counselling_data44.json", "r", encoding="utf-8") as f1:
        scraped = json.load(f1)
except FileNotFoundError as e:
    st.error(f"Data file missing: {e}")
    st.stop()


def chunk_text(text, max_tokens=384, overlap_tokens=64):
    """
    Token-based chunking using the BERT tokenizer (bert-large-uncased-whole-word-masking-finetuned-squad).
    Previously this used len(str.split()) which counts whitespace words, not subword tokens.
    A 250-word paragraph can easily exceed 300+ BERT subwords, causing silent truncation
    in bert_answer() when question + context exceed 512 tokens.
    Now we count real subword tokens and use a 64-token sliding-window overlap so no
    context is lost at chunk boundaries.
    """
    import re
    # Lazy-load the same tokenizer class + checkpoint used by bert_answer;
    # BertTokenizer is already imported at the top of this file.
    if not hasattr(chunk_text, "_tokenizer"):
        chunk_text._tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-cased-distilled-squad"
        )
    tokenizer = chunk_text._tokenizer

    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_ids = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_ids = tokenizer.encode(sentence, add_special_tokens=False)

        # Flush current chunk if adding this sentence would exceed max_tokens
        if len(current_ids) + len(sent_ids) > max_tokens and current_ids:
            chunks.append(
                tokenizer.decode(current_ids, skip_special_tokens=True).strip()
            )
            # Sliding-window overlap: carry last overlap_tokens to next chunk
            if len(current_ids) >= overlap_tokens:
                current_ids = current_ids[-overlap_tokens:]

        current_ids.extend(sent_ids)

    if current_ids:
        chunks.append(
            tokenizer.decode(current_ids, skip_special_tokens=True).strip()
        )

    return chunks


index = None
contexts = []
context_meta = []

try:
    with open("contexts_cache.pkl", "rb") as f:
        contexts, context_meta = pickle.load(f)
    index = faiss.read_index("faiss_index.bin")
except (FileNotFoundError, pickle.PickleError, Exception):
    contexts = []
    context_meta = []
    for item in scraped:
        if "content" in item:
            chunks = chunk_text(item["content"])
            contexts.extend(chunks)
            context_meta.extend([{"title": item["title"], "url": item["url"]}] * len(chunks))



try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

    if index is None:
        context_embeddings = embedder.encode(contexts, convert_to_tensor=True)
        emb_np = context_embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb_np)
        index = faiss.IndexFlatIP(emb_np.shape[1])
        index.add(emb_np)
        faiss.write_index(index, "faiss_index.bin")
        with open("contexts_cache.pkl", "wb") as f:
            pickle.dump((contexts, context_meta), f)

    from rank_bm25 import BM25Okapi
    tokenized_contexts = [c.split() for c in contexts]
    bm25 = BM25Okapi(tokenized_contexts)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()



def get_best_context(question):
    q_emb = embedder.encode(question, convert_to_tensor=True).cpu().numpy().astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)

    k = len(contexts)
    d_scores, d_idx = index.search(q_emb, k)
    d_scores = d_scores[0]
    d_idx = d_idx[0]

    dense_all = np.zeros(len(contexts), dtype="float32")
    dense_all[d_idx] = d_scores

    tokenized_query = question.split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype="float32")

    d_max = dense_all.max()
    dense_norm = dense_all / d_max if d_max > 0 else dense_all
    b_max = bm25_scores.max()
    bm25_norm = bm25_scores / b_max if b_max > 0 else bm25_scores

    fused = 0.6 * dense_norm + 0.4 * bm25_norm
    top5 = np.argsort(fused)[-5:][::-1]
    best_score = float(fused[top5[0]])

    if best_score < 0.65:
        return None
    top_contexts = [contexts[i] for i in top5]
    top_metas = [context_meta[i] for i in top5]
    return top_contexts, best_score, top_metas

def bert_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask
        )
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1

        # BUG 2 FIX: argmax can flip start/end; enforce at least one token
        if end <= start:
            end = start + 1

        # BUG 3 FIX: decode cleans up ## fragments and strips [CLS]/[SEP]
        answer = tokenizer.decode(input_ids[0][start:end], skip_special_tokens=True)
        return answer.strip()


st.set_page_config(page_title="IITK Chatbot", page_icon="🎓", layout="centered")

st.markdown(
    '''
    <h1 style='text-align: center; color: #004080;'> IITK Chatbot</h1>
    <p style='text-align: center; font-size:18px; color: gray;'>Ask anything related to <strong>IIT Kanpur</strong> – academics, hostels, mental health, orientation, reporting, and more!</p>
    ''', 
    unsafe_allow_html=True
)

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message["confidence"] is not None:
            confidence_pct = int(round(message["confidence"] * 100))
            st.markdown(f"<div style='margin-top: 8px; font-size: 12px; color: #888;'>Confidence: {confidence_pct}%</div>", unsafe_allow_html=True)
        if message["role"] == "assistant" and message.get("source"):
            source = message["source"]
            st.markdown(f"<div style='margin-top: 4px; font-size: 12px; color: #666;'>Source: <a href='{source['url']}' target='_blank'>{source['title']}</a></div>", unsafe_allow_html=True)

if user_query := st.chat_input("Ask your question here:"):
    st.session_state.messages.append({"role": "user", "content": user_query, "confidence": None, "source": None})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        result = get_best_context(user_query)

        if result is None:
            answer = "I'm not confident about that. Try rephrasing your question."
            confidence = None
            source = None
        else:
            top_contexts, score, top_metas = result
            rerank_scores = reranker.predict([(user_query, ctx) for ctx in top_contexts])
            best_idx = int(np.argmax(rerank_scores))
            best_ctx = top_contexts[best_idx]
            answer = bert_answer(user_query, best_ctx)
            confidence = score
            source = top_metas[best_idx]

    st.session_state.messages.append({"role": "assistant", "content": answer, "confidence": confidence, "source": source})

    with st.chat_message("assistant"):
        st.write(answer)
        if confidence is not None:
            confidence_pct = int(round(confidence * 100))
            st.markdown(f"<div style='margin-top: 8px; font-size: 12px; color: #888;'>Confidence: {confidence_pct}%</div>", unsafe_allow_html=True)
        if source:
            st.markdown(f"<div style='margin-top: 4px; font-size: 12px; color: #666;'>Source: <a href='{source['url']}' target='_blank'>{source['title']}</a></div>", unsafe_allow_html=True)

