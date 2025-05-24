import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load XGBoost model and Sentence-BERT encoder
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("xgboost_model.joblib")
    encoder = joblib.load("sentence_encoder.joblib")
    return model, encoder

model, encoder = load_model()

st.title("SmartCVMatcher – Matching CVs with a Job Offer")

job_desc = st.text_area("Enter the job description", height=150)

uploaded_files = st.file_uploader(
    "Upload multiple CVs (text files .txt)", 
    accept_multiple_files=True, 
    type=["txt"]
)

if st.button("Analyze the CVs"):
    if not job_desc:
        st.error("Please enter the job description")
    elif not uploaded_files:
        st.error("Please upload at least one CV")
    else:
        with st.spinner("Calculating matches..."):
            # Job description embedding
            job_emb = encoder.encode([job_desc])[0]  # shape: (768,)

            scores = []
            cv_texts = []

            for file in uploaded_files:
                text = file.read().decode("utf-8")
                cv_texts.append(text)

                # Resume embedding
                cv_emb = encoder.encode([text])[0]  # shape: (768,)

                # Concatenation (Resume + Job Offer)
                features = np.hstack([cv_emb, job_emb]).reshape(1, -1)  # shape: (1, 1536)

                # Prediction with XGBoost
                score = model.predict_proba(features)[0][1]  # Probability of class 1 (match)
                scores.append(score)

            df = pd.DataFrame({
                "CV": [f.name for f in uploaded_files],
                "Matching score": scores,
                "CV extract": [cv[:200].replace('\n', ' ') + "..." for cv in cv_texts]
            }).sort_values(by="Match score", ascending=False)

            st.write("### Ranking CVs by relevance :")
            st.dataframe(df.style.format({"Match score": "{:.3f}"}))