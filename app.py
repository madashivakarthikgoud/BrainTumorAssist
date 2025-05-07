import os
import cv2
import uuid
import base64
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
import time
from tensorflow.keras.optimizers import Adam
import skimage.transform as trans
from io import BytesIO

# Project imports
from CNN.model import cnn_bt
from ImageSegmentation.model_bt import unet_bt

# Constants
MAX_HISTORY = 10
PIXEL_TO_MM = 0.25  # 0.25mm per pixel

# Modern UI Theme
st.markdown("""
<style>
:root {
    --primary: #000000;
    --secondary: #ffffff;
    --accent: #ff4d4d;
    --success: #00b894;
    --background: #000000;
    --text: #ffffff;
}

.stApp {
    background: var(--background) !important;
    color: var(--text) !important;
}

.uploader-container {
    border: 2px dashed var(--text);
    border-radius: 12px;
    padding: 4rem 1rem;
    margin: 2rem 0;
    transition: all 0.3s ease;
    text-align: center;
}

.uploader-container:hover {
    border-color: var(--accent);
}

.metric-card {
    padding: 1.5rem;
    margin: 1rem 0;
}

.history-item {
    display: grid;
    grid-template-columns: 100px 1fr auto;
    gap: 1.5rem;
    align-items: center;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.2s ease;
}

.progress-bar {
    height: 6px;
    border-radius: 3px;
    background: rgba(255,255,255,0.1);
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.6s ease;
}

.healthy-progress {
    background: var(--success) !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-animation {
    animation: fadeIn 0.4s ease-out;
}
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_WEIGHTS = os.path.join(BASE_DIR, "CNN", "Model", "weights.hdf5")
UNET_WEIGHTS = os.path.join(BASE_DIR, "ImageSegmentation", "Model", "weights.hdf5")

def main():
    st.markdown("""
    <div style="max-width: 1200px; margin: 0 auto;">
        <h1 style="font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem;">
            NEUROVISION
        </h1>
        <div style="text-align: center; color: var(--accent); margin-bottom: 2rem;">
            AI-Powered Brain Analysis System
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 'scan_history' not in st.session_state:
        st.session_state.scan_history = []

    main_col = st.columns([1, 3], gap="large")
    
    # Sidebar Controls
    with main_col[0]:
        st.button("🧹 Clear History", use_container_width=True,
                 on_click=lambda: st.session_state.update(scan_history=[]))
        
        if st.download_button(
            "📥 Export Data", 
            data=generate_zip(st.session_state.scan_history),
            file_name="neurovision_export.zip",
            use_container_width=True,
            mime="application/zip"
        ):
            st.toast("Data exported successfully!")

    # Main Content
    with main_col[1]:
        uploaded_file = st.file_uploader(
            " ",
            type=["jpg", "jpeg", "png"], 
            help="Upload MRI scan for analysis",
            label_visibility="collapsed"
        )

        if uploaded_file:
            process_and_display(uploaded_file)
        
        display_history()

def process_and_display(uploaded_file):
    try:
        start_time = time.time()
        
        with st.spinner("🔍 Analyzing MRI Scan..."):
            img = Image.open(uploaded_file)
            img_array = np.array(img.convert('RGB'))
            orig_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            rows, cols = orig_gray.shape

            # Model predictions
            norm = orig_gray.astype("float32") / 255.0
            resized = trans.resize(norm, (256, 256)).reshape(1, 256, 256, 1)
            
            cnn_model, unet_model = load_models()
            pred_prob = cnn_model.predict(resized, verbose=0)[0][0]
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "original": img_array,
                "probability": float(pred_prob),
                "processing_time": time.time() - start_time,
                "dimensions": f"{cols}x{rows}",
                "file_size": f"{uploaded_file.size/1024:.1f}KB",
                "uid": str(uuid.uuid4())
            }

            if pred_prob > 0.5:
                # Tumor segmentation
                pred_mask = unet_model.predict(resized, verbose=0)[0, :, :, 0]
                mask_resized = trans.resize(pred_mask, (rows, cols), order=0, preserve_range=True)
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Create visualization with red overlay
                overlay = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
                overlay[mask_binary == 255] = [0, 0, 255]  # Red in BGR
                blended = cv2.addWeighted(cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)
                result["visualization"] = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                
                # Calculate metrics
                tumor_area_px = np.sum(mask_binary) / 255
                result["metrics"] = {
                    "area_px": tumor_area_px,
                    "area_mm2": tumor_area_px * (PIXEL_TO_MM ** 2),
                    "coverage": (tumor_area_px / (rows * cols)) * 100
                }

            # Update history
            st.session_state.scan_history = [result] + st.session_state.scan_history[:MAX_HISTORY-1]

        # Display results
        with st.container():
            st.markdown('<div class="result-animation">', unsafe_allow_html=True)
            with st.expander(f"📄 Analysis Report - {result['timestamp'][:19]}", expanded=True):
                if result.get('metrics'):
                    show_tumor_analysis(result)
                else:
                    show_healthy_analysis(result)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Analysis Error: {str(e)}")
        st.exception(e)

def show_tumor_analysis(result):
    cols = st.columns([1, 1], gap="large")
    with cols[0]:
        st.image(result['original'], use_container_width=True,
                caption=f"Original Scan • {result['dimensions']}")
    with cols[1]:
        st.image(result['visualization'], use_container_width=True,
                caption="Tumor Segmentation (Red Areas)")

    # Metrics
    m_cols = st.columns(3)
    with m_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: var(--accent); margin-bottom: 0.5rem;">TUMOR PROBABILITY</div>
            <div style="font-size: 2rem; font-weight: 700;">{result['probability']*100:.1f}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {result['probability']*100:.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with m_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: var(--accent); margin-bottom: 0.5rem;">AREA SIZE</div>
            <div style="font-size: 2rem; font-weight: 700;">{result['metrics']['area_mm2']:.1f} mm²</div>
            <div style="opacity: 0.8; margin-top: 0.5rem;">
                ({result['metrics']['area_px']:.0f} pixels)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with m_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: var(--accent); margin-bottom: 0.5rem;">TISSUE COVERAGE</div>
            <div style="font-size: 2rem; font-weight: 700;">{result['metrics']['coverage']:.1f}%</div>
            <div style="opacity: 0.8; margin-top: 0.5rem;">
                of total scan area
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_healthy_analysis(result):
    cols = st.columns([1, 1], gap="large")
    with cols[0]:
        st.image(result['original'], use_container_width=True,
                caption=f"Original Scan • {result['dimensions']}")
    
    with cols[1]:
        healthy_confidence = (1 - result['probability']) * 100
        st.markdown(f"""
        <div style="padding: 2rem; text-align: center;">
            <div style="font-size: 2rem; color: var(--success); margin-bottom: 1rem;">
                HEALTHY SCAN
            </div>
            <div style="font-size: 3rem; font-weight: 900; margin-bottom: 2rem;">
                {healthy_confidence:.1f}%
            </div>
            <div class="progress-bar">
                <div class="progress-fill healthy-progress" style="width: {healthy_confidence:.1f}%;"></div>
            </div>
            <div style="opacity: 0.8; margin-top: 1.5rem;">
                Confidence in healthy diagnosis
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_history():
    if st.session_state.scan_history:
        st.markdown("### Analysis History")
        for scan in st.session_state.scan_history:
            current_uid = scan['uid']  # Capture UID at iteration time
            with st.container():
                cols = st.columns([1, 4, 1])
                with cols[0]:
                    st.image(image_to_url(scan['original']), use_container_width=True)
                
                with cols[1]:
                    confidence = scan['probability']*100 if scan['probability'] > 0.5 else (1 - scan['probability'])*100
                    status_text = "TUMOR DETECTED" if scan['probability'] > 0.5 else "HEALTHY"
                    color = "var(--accent)" if scan['probability'] > 0.5 else "var(--success)"
                    
                    st.markdown(f"""
                    <div>
                        <div style="font-size: 1.2rem; color: {color};">
                            {status_text}
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">
                            {confidence:.1f}%
                        </div>
                        <div style="opacity: 0.8;">
                            {scan['timestamp'][:10]} • {scan['dimensions']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    # Use a callback with the current UID to ensure correct deletion
                    st.button(
                        "🗑", 
                        key=f"delete_{current_uid}", 
                        help="Delete this analysis",
                        on_click=delete_scan,
                        args=(current_uid,),
                    )

def delete_scan(uid_to_delete):
    """Callback function to delete a scan by UID"""
    st.session_state.scan_history = [
        s for s in st.session_state.scan_history 
        if s['uid'] != uid_to_delete
    ]

def generate_zip(history):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for idx, scan in enumerate(history):
            img = Image.fromarray(scan['original'])
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            zip_file.writestr(f"scan_{idx}_original.png", img_bytes.getvalue())
            
            report = f"""NeuroVision Analysis Report
            ---------------------------
            Date: {scan['timestamp'][:10]}
            File Size: {scan['file_size']}
            Dimensions: {scan['dimensions']}
            Tumor Probability: {scan['probability']*100:.1f}%"""
            
            if 'metrics' in scan:
                report += f"""
                Tumor Area: {scan['metrics']['area_mm2']:.2f} mm²
                Tissue Coverage: {scan['metrics']['coverage']:.2f}%"""
            
            zip_file.writestr(f"scan_{idx}_report.txt", report)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

@st.cache_resource
def load_models():
    try:
        cnn_model = cnn_bt(pretrained_weights=CNN_WEIGHTS)
        unet_model = unet_bt(pretrained_weights=UNET_WEIGHTS)
        unet_model.compile(
            optimizer=Adam(learning_rate=1e-4), 
            loss='binary_crossentropy'
        )
        return cnn_model, unet_model
    except Exception as e:
        st.error(f"🚨 Model Initialization Error: {str(e)}")
        st.stop()

def image_to_url(array):
    buffered = BytesIO()
    Image.fromarray(array).save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

if __name__ == '__main__':
    main()