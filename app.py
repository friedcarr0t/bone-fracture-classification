"""
Aplikasi Web untuk Klasifikasi Bone Fracture
Menggunakan Streamlit
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import os

# Page config
st.set_page_config(
    page_title="Bone Fracture Classification",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: rgba(60, 60, 60, 0.7);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-top: 1rem;
        color: #ffffff;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .prediction-box h3 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .prediction-box h2 {
        color: #ffffff !important;
        margin: 0.5rem 0;
        font-weight: 700;
        font-size: 1.5rem;
    }
    .prediction-box p {
        color: #ffffff;
        margin-top: 0.5rem;
        font-weight: 500;
        font-size: 1rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    /* Styling untuk tombol prediksi - warna biru sama dengan progress bar */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #1f77b4 !important;
        border-color: #1f77b4 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1565c0 !important;
        border-color: #1565c0 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:focus {
        background-color: #1f77b4 !important;
        border-color: #1f77b4 !important;
        box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25) !important;
    }
    /* Styling untuk progress bar text agar lebih mudah dibaca */
    div[data-testid="stProgress"] > div > div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    /* Styling untuk header dan subheader */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    /* Styling untuk teks umum agar lebih mudah dibaca */
    .stMarkdown, .stText {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model dengan caching"""
    NUM_CLASSES = 10
    
    def build_fusion_model(input_shape=(256, 256, 3), num_classes=10):
        inputs = layers.Input(shape=input_shape)
        
        # EfficientNet branch
        efficientnet = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        for layer in efficientnet.layers[:100]:
            layer.trainable = False
        
        x_eff = efficientnet.output
        x_eff = layers.GlobalAveragePooling2D()(x_eff)
        x_eff = layers.BatchNormalization()(x_eff)
        x_eff = layers.Dense(256, activation='relu')(x_eff)
        x_eff = layers.Dropout(0.3)(x_eff)
        
        # Transformer branch
        x_vit = layers.Conv2D(
            filters=128,
            kernel_size=(16, 16),
            strides=(8, 8),
            padding="same"
        )(inputs)

        _, h, w, c = x_vit.shape
        x_vit = layers.Reshape((h * w, c))(x_vit)
        
        pos_embed = layers.Embedding(
            input_dim=h * w,
            output_dim=c
        )(tf.range(start=0, limit=h * w, delta=1))
        x_vit = x_vit + pos_embed
        
        x_vit = layers.LayerNormalization(epsilon=1e-6)(x_vit)
        attn_output = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64
        )(x_vit, x_vit)
        x_vit = layers.Add()([attn_output, x_vit])
        
        x_vit = layers.LayerNormalization(epsilon=1e-6)(x_vit)
        attn_output = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64
        )(x_vit, x_vit)
        x_vit = layers.Add()([attn_output, x_vit])

        x_vit = layers.LayerNormalization(epsilon=1e-6)(x_vit)
        x_vit = layers.Dense(256, activation=tf.nn.gelu)(x_vit)
        x_vit = layers.Dropout(0.1)(x_vit)
        x_vit = layers.Dense(128, activation=tf.nn.gelu)(x_vit)
        
        x_vit = layers.GlobalAveragePooling1D()(x_vit)
        x_vit = layers.BatchNormalization()(x_vit)
        x_vit = layers.Dense(256, activation='relu')(x_vit)
        x_vit = layers.Dropout(0.3)(x_vit)
        
        # Fusion
        fusion = layers.Concatenate()([x_eff, x_vit])
        fusion = layers.BatchNormalization()(fusion)
        fusion = layers.Dense(512, activation='relu')(fusion)
        fusion = layers.Dropout(0.5)(fusion)
        fusion = layers.Dense(256, activation='relu')(fusion)
        fusion = layers.Dropout(0.3)(fusion)
        
        outputs = layers.Dense(num_classes, activation='softmax')(fusion)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    # Build model
    model = build_fusion_model(num_classes=NUM_CLASSES)
    
    # Load weights
    model_path = "bone_break_cnn_model.keras"
    if os.path.exists(model_path):
        model.load_weights(model_path)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        st.error(f"Model file tidak ditemukan: {model_path}")
        return None
    
    return model

@st.cache_data
def load_class_names():
    """Load class names dengan caching"""
    try:
        with open("class_names.json", "r") as f:
            return json.load(f)
    except:
        return ['Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation', 
                'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture', 
                'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture', 
                'Spiral Fracture']

def preprocess_image(img):
    """Preprocess image untuk model"""
    # Convert to RGB if needed (handle RGBA/grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    return img_preprocessed

def predict_fracture(model, img):
    """Prediksi jenis fraktur"""
    img_preprocessed = preprocess_image(img)
    predictions = model.predict(img_preprocessed, verbose=0)
    return predictions[0]

def get_confidence_class(confidence):
    """Kategori confidence"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">Sistem Klasifikasi Bone Fracture</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Informasi")
        st.markdown("""
        **Model:** Fusion CNN-Transformer  
        **Accuracy:** 88.57% (Test Set)  
        **Kelas:** 10 jenis fraktur
        """)
        
        st.markdown("---")
        st.header("Cara Penggunaan")
        st.markdown("""
        1. Upload gambar X-Ray tulang
        2. Klik tombol "Prediksi"
        3. Lihat hasil klasifikasi
        """)
        
        st.markdown("---")
        st.header("Kelas Fraktur")
        class_names = load_class_names()
        for i, name in enumerate(class_names, 1):
            st.markdown(f"{i}. {name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar X-Ray tulang",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar X-Ray dalam format JPG, JPEG, atau PNG"
        )
        
        if uploaded_file is not None:
            # Display image
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang diupload", use_container_width=True)
            
            # Prediction button
            if st.button("Prediksi", type="primary", use_container_width=True):
                with st.spinner("Memproses gambar..."):
                    # Load model
                    model = load_model()
                    
                    if model is not None:
                        # Predict
                        predictions = predict_fracture(model, img)
                        predicted_idx = np.argmax(predictions)
                        confidence = predictions[predicted_idx] * 100
                        class_names = load_class_names()
                        predicted_class = class_names[predicted_idx]
                        
                        # Store in session state
                        st.session_state['prediction'] = {
                            'class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': predictions,
                            'class_names': class_names
                        }
    
    with col2:
        st.header("Hasil Prediksi")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            
            # Main prediction box
            confidence_class = get_confidence_class(pred['confidence'] / 100)
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin-bottom: 0.5rem;">Jenis Fraktur Terdeteksi:</h3>
                <h2 style="margin: 0.5rem 0;">{pred['class']}</h2>
                <p style="margin-top: 0.5rem;">Confidence: <span class="{confidence_class}">{pred['confidence']:.2f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bars untuk semua kelas
            st.subheader("Probabilitas Semua Kelas")
            for i, (class_name, prob) in enumerate(zip(pred['class_names'], pred['all_predictions'])):
                prob_percent = float(prob) * 100
                prob_float = float(prob)  # Convert to Python float
                st.progress(prob_float, text=f"{class_name}: {prob_percent:.2f}%")
        else:
            st.info("Upload gambar dan klik tombol 'Prediksi' untuk melihat hasil")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Sistem Klasifikasi Otomatis Jenis Fraktur Tulang</strong></p>
        <p>Berbasis Deep Learning dengan Arsitektur Fusion CNN-Transformer</p>
        <p>Accuracy: 88.57% | Test Set: 140 images</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

