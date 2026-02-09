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
    
    /* Text section */
    .text-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .transcript-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2c3e50;
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
                    english_text = result.get('english_text', '')
                    threat_analysis = result.get('threat_analysis', {})
                    audio_duration = result.get('audio_duration', 0)
                    chunks_processed = result.get('chunks_processed', 1)
                    
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
                    # TRANSLATED TEXT
                    # ═══════════════════════════════════════════════
                    
                    st.markdown("""
                    <div class="text-section">
                        <div class="section-title">📝 Translated Text (English)</div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="transcript-text">{english_text}</p>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════
                    # METADATA
                    # ═══════════════════════════════════════════════
                    
                    st.markdown(f"""
                    <div class="metadata">
                        <div class="metadata-item">
                            <div class="metadata-label">Audio Duration</div>
                            <div class="metadata-value">{audio_duration:.1f}s</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Chunks Processed</div>
                            <div class="metadata-value">{chunks_processed}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Threat Level</div>
                            <div class="metadata-value">{threat_severity.upper()}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Processing completed successfully!")
                    
                else:
                    # Error from API
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                    error_msg = error_data.get('detail', 'Unknown error occurred')
                    
                    st.error(f"❌ Error: {error_msg}")
                    st.json(error_data)
                    
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