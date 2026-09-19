import io
import os
import streamlit as st
from PIL import Image
from gtts import gTTS
from huggingface_hub import InferenceClient
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

# ---------------------------------------------------------
# Page Configuration & Child-Friendly UI Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Magic Storybook AI",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        color: #FF5A5F;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model Loaders (Cached with st.cache_resource)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Image Captioning Model...")
def load_caption_pipeline():
    """Load BLIP processor and model directly."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def generate_image_caption(image: Image.Image, caption_pipe) -> str:
    """Generate image caption using BLIP processor and model."""
    processor, model = caption_pipe
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()
    )


# ---------------------------------------------------------
# Core Functional Blocks
# ---------------------------------------------------------
def generate_image_caption(image: Image.Image, caption_pipe) -> str:
    """Extract a brief natural language description from an image."""
    result = caption_pipe(image, max_new_tokens=50)
    return result[0]["generated_text"].strip()


def generate_kids_story(caption: str, story_pipe) -> str:
    """Generate a 50-100 word child-friendly bedtime story based on the caption."""
    prompt = (
        f"Write a cheerful bedtime story for kids aged 3 to 10 years old between 50 and 100 words. "
        f"The story must be based on this scene: {caption}. "
        f"Include a positive lesson and a happy ending."
    )
    story_result = story_pipe(
        prompt,
        max_length=200,
        min_length=60,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    return story_result[0]["generated_text"].strip()


def convert_text_to_audio(text: str) -> io.BytesIO:
    """Convert story text to speech using gTTS and return an in-memory byte buffer."""
    audio_buffer = io.BytesIO()
    tts = gTTS(text=text, lang='en', slow=False)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


def generate_video_from_text(prompt: str, hf_token: str):
    """
    Call Hugging Face Serverless Inference API to generate video from text.
    Preserves low RAM usage on Streamlit Cloud.
    """
    client = InferenceClient(provider="hf-inference", api_key=hf_token)
    video_bytes = client.text_to_video(
        prompt=prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )
    return video_bytes


# ---------------------------------------------------------
# Application Interface & Workflow
# ---------------------------------------------------------
def main():
    st.markdown('<div class="main-title">✨ Magic Storybook AI ✨</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Turn any picture into an interactive adventure story with audio and animation!</div>',
        unsafe_allow_html=True
    )

    # Sidebar for API keys and extra configurations
    with st.sidebar:
        st.header("⚙️ Settings")
        hf_token = st.text_input(
            "Hugging Face Token (Required for Video)",
            type="password",
            help="Free token from huggingface.co/settings/tokens to enable Text-to-Video generation."
        )
        st.markdown("---")
        st.info("💡 **Tip**: Models are cached to ensure fast responses without reloading.")

    caption_pipe = load_caption_pipeline()
    story_pipe = load_story_pipeline()

    # Image upload input
    uploaded_file = st.file_uploader(
        "Choose an image (PNG, JPG, JPEG)...",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_out = st.columns([1, 1], gap="medium")

        with col_img:
            st.subheader("📸 Your Image")
            st.image(image, use_container_width=True)

        with col_out:
            st.subheader("📖 Story & Narration")

            with st.spinner("Analyzing image..."):
                caption = generate_image_caption(image, caption_pipe)

            st.markdown(f"**Scene Description:** *{caption.capitalize()}*")

            with st.spinner("Weaving a story for ages 3-10..."):
                story = generate_kids_story(caption, story_pipe)

            st.success("Story ready!")
            st.write(story)

            # Word count verification indicator
            word_count = len(story.split())
            st.caption(f"📏 Length: {word_count} words (Target: 50–100 words)")

            # Audio Player
            st.markdown("### 🎧 Listen to Story")
            with st.spinner("Generating audio narration..."):
                audio_stream = convert_text_to_audio(story)
                st.audio(audio_stream, format="audio/mp3")

        # Text-to-Video Section
        st.markdown("---")
        st.subheader("🎬 Magic Video Animation")
        st.write("Generate a mini-clip based on your story scene.")

        if st.button("Generate Video Clip"):
            if not hf_token:
                st.warning("Please provide a Hugging Face API Token in the sidebar to run Text-to-Video.")
            else:
                with st.spinner("Generating video via Hugging Face Serverless API (this may take 30-60s)..."):
                    try:
                        # Extract first 25 words for a concise video prompt
                        video_prompt = f"cartoon style, {caption}, cinematic 3d render"
                        video_bytes = generate_video_from_text(video_prompt, hf_token)
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"Video generation failed: {e}")


if __name__ == "__main__":
    main()
