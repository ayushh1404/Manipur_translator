"""
Manipuri Audio Translation & Threat Detection Interface
Streamlit frontend for the FastAPI backend
"""

import streamlit as st
import requests
import os
from pathlib import Path
import time

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Manipuri Audio Translator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Upload section */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .threat-safe {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-left: 5px solid #28a745;
    }
    
    .threat-danger {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-left: 5px solid #dc3545;
    }
    
    /* Text section boxes - Dynamic sizing */
    .text-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 2px solid #e0e0e0;
        min-height: 100px;
        max-height: 500px;
        overflow-y: auto;
        transition: all 0.3s ease;
    }
    
    .text-box:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        border-color: #667eea;
    }
    
    .manipuri-box {
        border-left: 5px solid #ff6b6b;
        background: linear-gradient(to right, #fff5f5 0%, #ffffff 100%);
    }
    
    .english-box {
        border-left: 5px solid #667eea;
        background: linear-gradient(to right, #f0f4ff 0%, #ffffff 100%);
    }
    
    .section-label {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    
    .manipuri-label {
        background: #ff6b6b;
    }
    
    .english-label {
        background: #667eea;
    }
    
    .text-content {
        font-size: 1.15rem;
        line-height: 1.9;
        color: #2c3e50;
        padding: 0.5rem 0;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .manipuri-text {
        font-family: 'Noto Sans Meetei Mayek', 'Arial Unicode MS', sans-serif;
        direction: ltr;
    }
    
    /* Scrollbar styling for text boxes */
    .text-box::-webkit-scrollbar {
        width: 8px;
    }
    
    .text-box::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .text-box::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .text-box::-webkit-scrollbar-thumb:hover {
        background: #5568d3;
    }
    
    /* Threat indicator */
    .threat-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .threat-safe-indicator {
        background: #28a745;
        color: white;
    }
    
    .threat-danger-indicator {
        background: #dc3545;
        color: white;
    }
    
    /* Metadata */
    .metadata {
        display: flex;
        justify-content: space-around;
        margin-top: 1.5rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .metadata-item {
        text-align: center;
    }
    
    .metadata-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.3rem;
    }
    
    .metadata-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Processing animation */
    .processing {
        text-align: center;
        padding: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader styling */
    .stFileUploader {
        background: white;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Backend API URL - change this to your FastAPI server URL
API_URL = os.getenv("API_URL", "http://localhost:8000")
ENDPOINT = f"{API_URL}/voice/manipuri-threat-check"

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-container">
    <div class="header-title">🎙️ Manipuri Audio Translator</div>
    <div class="header-subtitle">Speech-to-Text Translation with AI-Powered Threat Detection</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════════

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Audio File")
st.markdown("Upload your Manipuri audio file (MP4, WAV, MP3, or other formats)")

uploaded_file = st.file_uploader(
    "",
    type=['mp4', 'wav', 'mp3', 'webm', 'm4a', 'ogg'],
    help="Select an audio file in Manipuri language"
)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PROCESS BUTTON & RESULTS
# ═══════════════════════════════════════════════════════════════════

if uploaded_file is not None:
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        process_btn = st.button("🚀 Process Audio", use_container_width=True, type="primary")
    
    if process_btn:
        
        # Show processing animation
        with st.spinner('🔄 Processing your audio...'):
            
            try:
                # Prepare file for upload
                files = {
                    'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                # Make API request
                response = requests.post(ENDPOINT, files=files, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract data
                    manipur_text = result.get('manipur_text', '')
                    english_text = result.get('english_text', '')
                    threat_analysis = result.get('threat_analysis', {})
                    
                    is_threat = threat_analysis.get('threat', False)
                    threat_reason = threat_analysis.get('reason', '')
                    threat_severity = threat_analysis.get('severity', 'low')
                    
                    # ═══════════════════════════════════════════════
                    # THREAT INDICATOR
                    # ═══════════════════════════════════════════════
                    
                    if is_threat:
                        st.markdown(f"""
                        <div class="result-card threat-danger">
                            <div class="threat-indicator threat-danger-indicator">
                                🚨 THREAT DETECTED
                            </div>
                            <h3>Severity: {threat_severity.upper()}</h3>
                            <p style="font-size: 1.1rem; margin-top: 1rem;">
                                <strong>Reason:</strong> {threat_reason}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card threat-safe">
                            <div class="threat-indicator threat-safe-indicator">
                                ✅ NO THREAT DETECTED
                            </div>
                            <p style="font-size: 1.1rem; margin-top: 1rem;">
                                <strong>Analysis:</strong> {threat_reason}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════
                    # MANIPURI SCRIPT BOX
                    # ═══════════════════════════════════════════════
                    
                    if manipur_text:
                        # Calculate dynamic height based on text length
                        text_length = len(manipur_text)
                        if text_length < 200:
                            box_height = "auto"
                        elif text_length < 500:
                            box_height = "200px"
                        else:
                            box_height = "400px"
                        
                        st.markdown(f"""
                        <div class="text-box manipuri-box" style="max-height: {box_height};">
                            <span class="section-label manipuri-label">📜 Manipuri Script</span>
                            <div class="text-content manipuri-text">{manipur_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════
                    # ENGLISH TRANSLATION BOX
                    # ═══════════════════════════════════════════════
                    
                    if english_text:
                        # Calculate dynamic height based on text length
                        text_length = len(english_text)
                        if text_length < 200:
                            box_height = "auto"
                        elif text_length < 500:
                            box_height = "200px"
                        else:
                            box_height = "400px"
                        
                        st.markdown(f"""
                        <div class="text-box english-box" style="max-height: {box_height};">
                            <span class="section-label english-label">📝 English Translation</span>
                            <div class="text-content">{english_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════
                    # METADATA
                    # ═══════════════════════════════════════════════
                    
                    st.markdown(f"""
                    <div class="metadata">
                        <div class="metadata-item">
                            <div class="metadata-label">Threat Level</div>
                            <div class="metadata-value">{threat_severity.upper()}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Manipuri Text Length</div>
                            <div class="metadata-value">{len(manipur_text)}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">English Text Length</div>
                            <div class="metadata-value">{len(english_text)}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Processing completed successfully!")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. The audio file might be too long or the server is busy. Please try again.")
                
            except requests.exceptions.ConnectionError:
                st.error(f"🔌 Cannot connect to the API server at {API_URL}. Please ensure the backend is running.")
                
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.exception(e)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR INFO
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This application uses AI to:
    - Transcribe Manipuri audio to text
    - Translate to English
    - Detect potential threats
    
    **Supported formats:**
    - MP4, WAV, MP3, WebM, M4A, OGG
    
    **Processing:**
    - Short audio (<25s): Single pass
    - Long audio (>25s): Chunked processing
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 API Configuration")
    st.code(f"API URL: {API_URL}", language="text")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Backend Online")
        else:
            st.warning("⚠️ Backend Degraded")
    except:
        st.error("🔴 Backend Offline")

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Powered by Sarvam AI • OpenAI • FastAPI • Streamlit</p>
</div>
""", unsafe_allow_html=True)