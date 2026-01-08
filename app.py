import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import os

# Import shared utilities
from utils import (
    CLASSES,  # Traffic sign classes dictionary
    NUM_CLASSES,  # Total number of classes (43)
    IMG_HEIGHT,  # Image height (30)
    IMG_WIDTH,  # Image width (30)
    preprocess_image,  # Unified preprocessing function
    predict_top_classes,  # Unified prediction function
    get_image_shape  # Get (height, width, channels)
)


@st.cache_resource
def load_model(model_path):
    """Load the trained model (cached)"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Streamlit UI Configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .top-prediction {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚦 Traffic Sign Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload images of traffic signs to get instant predictions</p>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(f"""
    This application uses a Convolutional Neural Network (CNN) 
    to classify German Traffic Signs (GTSRB dataset).

    **Features:**
    - Upload single or multiple images
    - Get top 5 predictions with confidence scores
    - Supports PNG, JPG, JPEG formats

    **Model Info:**
    - {NUM_CLASSES} traffic sign classes
    - Input size: {IMG_HEIGHT}x{IMG_WIDTH} pixels
    - Architecture: Deep CNN with BatchNorm
    """)

    st.header("📊 Statistics")
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    st.metric("Total Predictions", st.session_state.total_predictions)

# Main content
model_path = os.getenv('MODEL_PATH', 'models/best_gtsrb_model.h5')

# Load model
with st.spinner('Loading model...'):
    model = load_model(model_path)

if model is None:
    st.error("❌ Failed to load model. Please check if the model file exists.")
    st.stop()

st.success("✅ Model loaded successfully!")

# File uploader
uploaded_files = st.file_uploader(
    "Choose traffic sign image(s)",
    type=['png', 'jpg', 'jpeg', 'ppm'],
    accept_multiple_files=True,
    help="Upload one or more images of traffic signs"
)

if uploaded_files:
    st.markdown("---")

    # Process each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### 🖼️ Image {idx + 1}: {uploaded_file.name}")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            # Preprocess and predict
            with st.spinner('Analyzing...'):
                try:
                    # Use shared preprocessing function - accepts PIL Image
                    processed_image = preprocess_image(image)

                    # Use shared prediction function
                    predictions = predict_top_classes(model, processed_image, top_n=5)

                    if predictions:
                        st.session_state.total_predictions += 1

                        # Display top prediction prominently
                        top_pred = predictions[0][0]
                        st.markdown(
                            f"""
                            <div class="prediction-box top-prediction">
                                <h3>🎯 Top Prediction</h3>
                                <h2>{top_pred['class_name']}</h2>
                                <h3>{top_pred['probability'] * 100:.2f}% confidence</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Display all top 5 predictions
                        st.markdown("#### 📊 Top 5 Predictions:")

                        for i, pred in enumerate(predictions[0]):
                            confidence = pred['probability'] * 100

                            # Create progress bar
                            col_name, col_bar, col_pct = st.columns([2, 3, 1])

                            with col_name:
                                st.write(f"**{i + 1}. {pred['class_name']}**")

                            with col_bar:
                                st.progress(pred['probability'])

                            with col_pct:
                                st.write(f"{confidence:.1f}%")

                except Exception as e:
                    st.error(f"❌ Error processing image: {e}")

        # Add separator between images
        if idx < len(uploaded_files) - 1:
            st.markdown("---")

else:
    # Show example/instructions when no files uploaded
    st.info("👆 Upload one or more images to get started!")

    with st.expander("📸 Example Traffic Signs"):
        st.write(f"""
        The model can recognize {NUM_CLASSES} different types of traffic signs including:
        - Speed limits (20-120 km/h)
        - Warning signs (curves, pedestrians, animals)
        - Mandatory signs (roundabout, keep right/left)
        - Prohibitory signs (no entry, no passing)
        - And many more!
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit 🎈 | Powered by TensorFlow 🧠</p>
    </div>
    """,
    unsafe_allow_html=True
)