#import part
import streamlit as st
from transformers import pipeline

#function part


#main part


# Setup page configuration
st.set_page_config(page_title="Deep Learning Sentiment Analysis App", page_icon="💩", layout="centered")

st.title("💩 Sentiment Analysis Tool💩")
st.write("Enter text to analyze")

# Cache the pipeline so the model loads only once
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

with st.spinner("Loading model..."):
    sentiment_pipeline = load_sentiment_model()

# Text input area with the default text from your notebook
default_text = "比卡超見記者"
user_input = st.text_area("Input Text:", value=default_text, height=150)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_pipeline(user_input)
            label = result[0]["label"]
            score = result[0]["score"]

        st.subheader("Result")
        if label.upper() == "POSITIVE":
            st.success(f"**Sentiment:** {label}")
        else:
            st.error(f"**Sentiment:** {label}")

        st.metric(label="Confidence Score", value=f"{score:.4f}")
    else:
        st.warning("Please enter some text to analyze.")
