import os
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# 1. Load Hugging Face Models
@st.cache_resource
def load_models():
    """
    Loads and caches the image-captioning and text-generation models.
    Models used:
      - Image Captioning: Salesforce/blip-image-captioning-base
      - Text Generation:  gpt2
    """
    caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_model = pipeline("text-generation", model="gpt2")
    return caption_model, story_model

# 2. Image Captioning Function
def get_caption(image, caption_pipe):
    """Generates a descriptive caption from an uploaded image."""
    result = caption_pipe(image)
    return result[0]["generated_text"]

# 3. Story Generation Function
def get_story(caption, story_pipe):
    """Expands the caption into a 50-100 word child-friendly story."""
    prompt = f"Once upon a time, there was {caption}. One sunny day,"
    output = story_pipe(
        prompt,
        max_new_tokens=80,
        min_length=50,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2
    )
    story = output[0]["generated_text"]
    # Trim to the final complete sentence
    return story[:story.rfind(".") + 1] if "." in story else story

# 4. Text-to-Speech Function
def text_to_speech(text, filename="story.mp3"):
    """Converts the generated story text into an audio file using gTTS."""
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# 5. Streamlit Web UI
def main():
    st.title("📖 Children's Story Generator")
    st.write("Upload an image to create an exciting bedtime story with audio!")

    caption_pipe, story_pipe = load_models()

    uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Story"):
            with st.spinner("Processing image and generating audio..."):
                # Step A: Image to text
                caption = get_caption(image, caption_pipe)
                st.info(f"**Caption:** {caption}")

                # Step B: Text generation
                story = get_story(caption, story_pipe)
                st.subheader("Your Story:")
                st.write(story)

                # Step C: Text-to-speech audio
                audio_path = text_to_speech(story)
                st.audio(audio_path, format="audio/mp3")

                # Cleanup temporary file
                if os.path.exists(audio_path):
                    os.remove(audio_path)

if __name__ == "__main__":
    main()
