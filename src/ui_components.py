"""
UI Components Module - Autonomous Candidate Screening Platform
Provides reusable UI overlays, loaders, and presentation elements.
"""

from contextlib import contextmanager
import streamlit as st

@contextmanager
def loading_screen(message: str = "Loading, please wait...", subtext: str = ""):
    """
    Displays a fullscreen semi-transparent dark backdrop overlay
    with a modern circular spinner and text indicator.
    """
    placeholder = st.empty()
    subtext_html = f'<p class="loading-subtext">{subtext}</p>' if subtext else ""
    
    loader_html = f"""
    <style>
    .loading-overlay-backdrop {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        z-index: 999999;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        animation: fadeInOverlay 0.25s ease-in-out;
    }}
    
    .circular-spinner {{
        width: 52px;
        height: 52px;
        border: 4px solid rgba(255, 255, 255, 0.15);
        border-top: 4px solid #38bdf8;
        border-right: 4px solid #818cf8;
        border-radius: 50%;
        animation: spinCircular 0.85s linear infinite;
        margin-bottom: 20px;
    }}
    
    .loading-text {{
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.3px;
    }}
    
    .loading-subtext {{
        color: #94a3b8;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 0.88rem;
        margin-top: 8px;
        margin-bottom: 0;
    }}
    
    @keyframes spinCircular {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    @keyframes fadeInOverlay {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    </style>
    
    <div class="loading-overlay-backdrop">
        <div class="circular-spinner"></div>
                    <p class="loading-text">{message}</p>
                    {subtext_html}
    </div>
    """
    
    placeholder.markdown(loader_html, unsafe_allow_html=True)
    try:
        yield placeholder
    finally:
        placeholder.empty()
