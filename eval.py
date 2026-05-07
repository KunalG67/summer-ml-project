import sys
sys.path.insert(0, ".")
from app import get_best_context, bert_answer, contexts, context_meta, index, bm25, reranker

def compute_f1(pred_tokens, true_tokens):
    pred_set = set(pred_tokens)
    true_set = set(true_tokens)
    common = pred_set.intersection(true_set)
    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(true_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

test_set = [
    ("What are the office timings of the Counselling Service?", "11:00 AM - 7:00 PM (Mon to Fri)"),
    ("What is the phone number of the Counselling Service?", "+ 91 512 2597784"),
    ("How can I email for an appointment?", "counselor@iitk.ac.in"),
    ("Who are the psychiatrists at IITK?", "Dr. Alok Bajpai, Dr. Sanjay Mahendru, Dr. Rohan Kumar"),
    ("How often does Dr. Alok Bajpai visit?", "weekly basis"),
    ("How long is the Orientation programme?", "11 - day"),
    ("When is World Suicide Prevention Day observed?", "10 September"),
    ("What is the SBF scholarship for?", "all the registered students who are not"),
    ("How do I apply for SBF scholarship?", "Office Automation ( OA ) Portal"),
    ("Which banks are available at IITK?", "State Bank of India, Union Bank of India, ICICI Bank"),
    ("What is the nearest airport to IIT Kanpur?", "Lucknow Airport"),
    ("How far is Kanpur Central Railway Station from IITK?", "17 km"),
    ("How many credits is a semester load?", "36"),
    ("Is counselling service free for students?", "free of cost"),
    ("What is the Hakuna Matata event?", "Diwali celebration"),
]

print(f"{'Question':<60} | {'Predicted':<40} | {'Expected':<30} | {'EM':<4} | {'F1':<5}")
print("-" * 150)

em_scores = []
f1_scores = []

for question, expected in test_set:
    result = get_best_context(question)
    if result is None:
        predicted = "I'm not confident about that. Try rephrasing your question."
    else:
        top_contexts, score, top_metas = result
        rerank_scores = reranker.predict([(question, ctx) for ctx in top_contexts])
        best_idx = int(__import__("numpy").argmax(rerank_scores))
        predicted = bert_answer(question, top_contexts[best_idx])
    
    pred_tokens = predicted.lower().split()
    true_tokens = expected.lower().split()
    
    em = 1 if predicted.strip() == expected.strip() else 0
    f1 = compute_f1(pred_tokens, true_tokens)
    
    em_scores.append(em)
    f1_scores.append(f1)
    
    print(f"{question[:60]:<60} | {predicted[:40]:<40} | {expected[:30]:<30} | {em:<4} | {f1:.2f}")

avg_em = sum(em_scores) / len(em_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

print("-" * 150)
print(f"Average EM: {avg_em:.2f}")
print(f"Average F1: {avg_f1:.2f}")
