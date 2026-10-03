import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

def main():
    st.title("yelp2024fall Test")
    st.write("Enter a sentence for analysis:")

    user_input = st.text_input("Review text:")
    
    if user_input:
        # Removed the invalid << >> brackets around the model name
        model2 = AutoModelForSequenceClassification.from_pretrained(
            "kktlau115/L5TutorialYELP", 
            num_labels=5
        )
        tokenizer = AutoTokenizer.from_pretrained("kktlau115/L5TutorialYELP")

        inputs = tokenizer(
            user_input,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = model2(**inputs)
            
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().numpy()
        
        max_index = np.argmax(predictions)
        st.write(f"Result (AutoModel) - Label: {max_index} (Star Rating: {max_index + 1})")

if __name__ == "__main__":
    main()
