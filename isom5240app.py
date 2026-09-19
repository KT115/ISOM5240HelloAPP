# import part
import streamlit as st
from transformers import pipeline


# function part
@st.cache_resource
def load_sentiment_model():
    # Explicitly defining the default model used in your Colab environment
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline("sentiment-analysis", model=model_name)

def analyze_text(sentiment_pipeline, text):
    # Process the text and extract label and score
    result = sentiment_pipeline(text)
    label = result[0]["label"]
    score = result[0]["score"]
    return label, score

def main():
    # Setup page configuration
    st.set_page_config(page_title="Deep Learning Sentiment Analysis App", page_icon="💩", layout="centered")

    st.title("💩 Sentiment Analysis Tool 💩")
    st.write("Enter text to analyze")

    # Load the model
    with st.spinner("Loading model..."):
        sentiment_pipeline = load_sentiment_model()

    # Text input area with the default text
    default_text = "比卡超見記者"
    user_input = st.text_area("Input Text:", value=default_text, height=150)

    # Trigger analysis on button click
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                label, score = analyze_text(sentiment_pipeline, user_input)

            st.subheader("Result")
            if label.upper() == "POSITIVE":
                st.success(f"**Sentiment:** {label}")
            else:
                st.error(f"**Sentiment:** {label}")

            st.metric(label="Confidence Score", value=f"{score:.4f}")
        else:
            st.warning("Please enter some text to analyze.")


# main part
if __name__ == "__main__":
    main()
