

import streamlit as st
import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer, util


with open("iitk_cleaned_qna22.json", "r", encoding="utf-8") as f2:
    qna = json.load(f2)
with open("iitk_counselling_data44.json", "r", encoding="utf-8") as f1:
    scraped = json.load(f1)


def chunk_text(text, max_tokens=250):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) <= max_tokens:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks


contexts = []
for item in scraped:
    if "content" in item:
        chunks = chunk_text(item["content"])
        contexts.extend(chunks)

contexts += [pair["answer"] for pair in qna if "answer" in pair]


embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


context_embeddings = embedder.encode(contexts, convert_to_tensor=True)



def get_best_context(question):
    query_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, context_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return contexts[best_idx]

def bert_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[0][start:end])
        )
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

user_query = st.text_input("🔎 Ask your question here:")
if user_query:
    with st.spinner("Thinking... "):
        context = get_best_context(user_query)
        answer = bert_answer(user_query, context)

        st.markdown(f"<b>You:</b> {user_query}", unsafe_allow_html=True)
        st.markdown(
            f'''
            <div style='background-color:#f0f8ff; padding: 15px 20px; border-radius: 10px; margin-top: 10px;'>
                <b style='color: #004080;'>ChatBot:</b>
                <span style='color: black;'>{answer}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )

#  Optional: Input Styling
st.markdown(
    '''
    <style>
    .stTextInput > div > div > input {
        border: 2px solid #004080;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

