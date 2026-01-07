import re
import pandas as pd
import streamlit as st
import joblib

def load_models():
    clf = joblib.load("logistic_regression.joblib")
    reg = joblib.load("ridge_regressor.joblib")
    return clf, reg
classifier, regressor = load_models()
def preprocess_input(desc, inp_desc, out_desc, sample_io):
    patterns = [
        r"sample input\s*\d*",
        r"sample output\s*\d*",
        r"input\s*\d*",
        r"output\s*\d*",
        r"example\s*\d*",
    ]
    for p in patterns:
        sample_io= re.sub(p,"",sample_io)
    sample_io = re.sub(r"\n+", "\n",sample_io)
    sample_io= re.sub(r"\s+", " ",sample_io) 
    raw_text=(desc + " " +inp_desc + " " +out_desc + " " +sample_io)
    raw_text=raw_text.lower()
    raw_text = re.sub(r"\n+", "\n", raw_text)
    raw_text=re.sub(r"\s+", " ", raw_text).strip()
    word_count=len(raw_text.split())

    return pd.DataFrame({
        "raw_text": [raw_text],
        "word_count": [word_count],
        "description_empty": [1 if desc.strip() == "" else 0]
    })


st.set_page_config(page_title="AutoJudge", layout="wide")
st.title("AutoJudge – Problem Difficulty Predictor")
st.divider()
st.subheader("**Enter problem details to predict **difficulty class** or **difficulty score****.")
st.divider()
col1, col2 = st.columns(2)

with col1:
    desc = st.text_area("Problem Description", height=200)
    inp_desc = st.text_area("Input Description", height=100)

with col2:
    out_desc = st.text_area("Output Description", height=100)
    sample_io = st.text_area("Sample Testcases", height=120)

st.divider()
colA, colB = st.columns(2)

with colA:
    if st.button("Predict Problem Class"):
        if not (desc.strip() or inp_desc.strip() or out_desc.strip() or sample_io.strp()):
            st.warning("Please enter at least one text field.")
            st.stop()

        else:
            df = preprocess_input(desc, inp_desc, out_desc, sample_io)
            pred_class = classifier.predict(df)[0]

            st.subheader("Predicted Difficulty Class")
            if pred_class == "easy":
                st.success("Easy")
            elif pred_class == "medium":
                st.warning("Medium")
            else:
                st.error("Hard")

with colB:
    if st.button("Predict Problem Score"):
        if not (desc.strip() or inp_desc.strip() or out_desc.strip() or sample_io.strip())  :
            st.warning("Please enter at least one text field.")
            st.stop()

        else:
            df = preprocess_input(desc, inp_desc, out_desc, sample_io)
            pred_score = regressor.predict(df)[0]

            st.subheader("Predicted Difficulty Score")
            st.metric("Score (1–10)", round(pred_score, 2))
            st.progress(min(pred_score / 10, 1.0))