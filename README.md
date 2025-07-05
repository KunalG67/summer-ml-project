 IITK Chatbot

This is a question-answering chatbot for IIT Kanpur students using BERT and Sentence Transformers.  
It can answer queries related to:
- Hostels 
- Mental Health Support
- Orientation & Reporting 
- Academics
- And more...

Working
- Uses BERT for answer extraction
- Uses SentenceTransformer to find the best context from IITK scraped data and FAQs
- UI built with Streamlit

Files
- `app.py` – Main Streamlit chatbot code
- `iitk_cleaned_qna22.json` – Cleaned QnA file
- `iitk_counselling_data44.json` – Scraped content from IITK counselling site

 How to Run
```bash
streamlit run app.py
