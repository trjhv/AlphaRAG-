"""
OmniFin RAG - Frontend Application
===================================

A professional financial document analysis interface built with Streamlit.
This frontend communicates with a secure backend API that handles all
proprietary AI processing, including:
- FAISS vector search over 10-K documents
- BM25 hybrid retrieval
- Groq LPU-powered ReAct agent reasoning
- Multi-step tool orchestration

Author: [Your Name]
Project: Final Year Project - AI-Powered Financial Analysis Platform
Architecture: Client-Server with REST API communication
"""

import streamlit as st
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Backend API endpoint (replace with your deployed backend URL)
BACKEND_URL = "http://localhost:8000"  # Development
# BACKEND_URL = "https://your-secure-backend.com"  # Production

# API timeout settings
UPLOAD_TIMEOUT = 120  # seconds
QUERY_TIMEOUT = 60    # seconds

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="OmniFin RAG - Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL GLASSMORPHISM UI STYLING
# ============================================================================

st.markdown("""
<style>
    /* Modern dark theme with glassmorphism effects */
    
    /* Hide Streamlit branding for professional appearance */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animated gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        animation: headerSlideIn 0.8s ease-out;
    }
    
    @keyframes headerSlideIn {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Glassmorphism card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.3);
    }
    
    /* Status badges with pulse animation */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.3rem;
        transition: all 0.3s;
    }
    
    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .status-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .status-badge.pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.9; }
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* ChatGPT-style message bubbles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: messageSlideIn 0.4s ease-out;
    }
    
    @keyframes messageSlideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-right: 20%;
    }
    
    /* Skeleton loading animation */
    .skeleton {
        background: linear-gradient(90deg, #2a2d3a 25%, #363945 50%, #2a2d3a 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 8px;
        height: 20px;
        margin: 10px 0;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .user-message, .assistant-message {
            margin-left: 0;
            margin-right: 0;
        }
        
        .stButton>button {
            width: 100%;
            margin: 5px 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """
    Initialize Streamlit session state for chat history and analytics.
    
    This manages the frontend state only. All AI processing state is
    maintained server-side in the secure backend.
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'total_queries': 0,
            'successful_queries': 0,
            'response_times': [],
            'query_history': []
        }
    
    if 'backend_connected' not in st.session_state:
        st.session_state.backend_connected = False
    
    if 'files_uploaded' not in st.session_state:
        st.session_state.files_uploaded = False

initialize_session_state()

# ============================================================================
# API COMMUNICATION LAYER
# ============================================================================

def check_backend_health() -> Optional[Dict]:
    """
    Check if the backend API is accessible and healthy.
    
    Returns:
        Dict with backend status or None if unreachable
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=5
        )
        if response.status_code == 200:
            st.session_state.backend_connected = True
            return response.json()
        return None
    except requests.exceptions.RequestException:
        st.session_state.backend_connected = False
        return None


def upload_files_to_backend(
    faiss_file,
    metadata_file,
    bm25_file,
    api_key: str
) -> Optional[Dict]:
    """
    Upload pre-processed document files to the secure backend.
    
    The backend performs:
    - FAISS index loading (768-dim vector embeddings)
    - Metadata reconstruction
    - BM25 tokenized corpus initialization
    - CrossEncoder model loading for reranking
    
    Args:
        faiss_file: FAISS vector index file
        metadata_file: Document metadata pickle file
        bm25_file: BM25 search index file
        api_key: Groq API key for LLM access
    
    Returns:
        Dict with upload status or None on failure
    """
    try:
        # Prepare multipart form data for file upload
        files = {
            'faiss_file': ('index.faiss', faiss_file.getvalue()),
            'metadata_file': ('metadata.pkl', metadata_file.getvalue()),
            'bm25_file': ('bm25.pkl', bm25_file.getvalue())
        }
        
        data = {'groq_api_key': api_key}
        
        # Send files to backend API
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files,
            data=data,
            timeout=UPLOAD_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.files_uploaded = True
            return result
        else:
            st.error(f"Upload failed: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Upload timeout. Files may be too large.")
        return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def analyze_query(
    query: str,
    company: str,
    ticker: str
) -> Optional[Dict]:
    """
    Send user query to backend for AI-powered analysis.
    
    Backend processing pipeline:
    1. Hybrid FAISS + BM25 search (70/30 weighted)
    2. CrossEncoder reranking (top-3 results)
    3. ReAct agent loop with Groq LPU (llama-3.3-70b)
    4. Multi-tool orchestration (RAG, stock data, calculations)
    5. Iterative reasoning (up to 5 iterations)
    
    Args:
        query: User's natural language question
        company: Company name for context
        ticker: Stock ticker symbol
    
    Returns:
        Dict with answer, reasoning steps, and metadata
    """
    try:
        # Construct API request payload
        payload = {
            "query": query,
            "company": company,
            "ticker": ticker
        }
        
        # Send query to backend for processing
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json=payload,
            timeout=QUERY_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Query timeout. Complex queries may take longer.")
        return None
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def show_skeleton_loader():
    """Display animated skeleton loader during API processing."""
    st.markdown("""
    <div class="skeleton" style="width: 60%; height: 25px;"></div>
    <div class="skeleton" style="width: 100%; height: 15px; margin-top: 15px;"></div>
    <div class="skeleton" style="width: 90%; height: 15px;"></div>
    <div class="skeleton" style="width: 95%; height: 15px;"></div>
    <div class="skeleton" style="width: 100%; height: 200px; margin-top: 20px;"></div>
    """, unsafe_allow_html=True)


def track_query_analytics(
    query: str,
    tools_used: List[str],
    response_time: float,
    success: bool
):
    """
    Track query analytics for performance monitoring.
    
    This is frontend-only tracking. Backend maintains separate
    audit logs for security and compliance.
    """
    st.session_state.analytics['total_queries'] += 1
    if success:
        st.session_state.analytics['successful_queries'] += 1
    
    st.session_state.analytics['response_times'].append(response_time)
    st.session_state.analytics['query_history'].append({
        'query': query,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tools': tools_used,
        'time': response_time,
        'success': success
    })

# ============================================================================
# MAIN APPLICATION UI
# ============================================================================

def main():
    """
    Main application entry point.
    
    Architecture:
    - Frontend: Streamlit UI (this file)
    - Backend: FastAPI server with proprietary AI logic
    - Communication: REST API with JSON payloads
    - Security: API key authentication, server-side validation
    """
    
    # Animated header
    st.markdown("""
    <div class="main-header">
        <h1>📊 OmniFin RAG</h1>
        <p>AI-Powered Financial Document Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check backend connectivity
    health_status = check_backend_health()
    
    # Sidebar - System Status
    with st.sidebar:
        st.markdown("### 🎯 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            backend_status = "success" if st.session_state.backend_connected else "warning"
            backend_icon = "✓" if st.session_state.backend_connected else "○"
            st.markdown(
                f'<span class="status-badge status-{backend_status}">{backend_icon} Backend</span>',
                unsafe_allow_html=True
            )
            
            files_status = "success" if st.session_state.files_uploaded else "info"
            files_icon = "✓" if st.session_state.files_uploaded else "○"
            st.markdown(
                f'<span class="status-badge status-{files_status}">{files_icon} Data</span>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown('<span class="status-badge status-success">✓ UI</span>', unsafe_allow_html=True)
            st.markdown('<span class="status-badge status-info">○ Analytics</span>', unsafe_allow_html=True)
        
        if not st.session_state.backend_connected:
            st.warning(f"⚠️ Backend offline. Start server at {BACKEND_URL}")
        
        st.markdown("---")
        
        # Company selector
        st.markdown("### 🏢 Company")
        companies = {
            'Apple': 'AAPL',
            'Microsoft': 'MSFT',
            'Alphabet': 'GOOGL',
            'Amazon': 'AMZN',
            'Tesla': 'TSLA'
        }
        
        selected_company = st.selectbox(
            "Select:",
            options=sorted(companies.keys()),
            label_visibility="collapsed"
        )
        company_ticker = companies[selected_company]
        
        # Analytics summary
        if st.session_state.analytics['total_queries'] > 0:
            st.markdown("---")
            st.markdown("### 📊 Session Stats")
            
            success_rate = (
                st.session_state.analytics['successful_queries'] / 
                st.session_state.analytics['total_queries'] * 100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Queries", st.session_state.analytics['total_queries'])
            with col2:
                st.metric("Success", f"{success_rate:.0f}%")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload Documents",
        "💬 AI Analysis Chat",
        "📈 Analytics Dashboard"
    ])
    
    # ========================================================================
    # TAB 1: Document Upload
    # ========================================================================
    
    with tab1:
        st.markdown("### 📤 Upload Pre-Processed Document Files")
        
        st.info("""
        **Required Files:**
        - 🔍 FAISS Index (.faiss) - Vector embeddings
        - 📊 Metadata (.pkl) - Document chunks
        - 📈 BM25 Index (.pkl) - Keyword search
        
        Files are securely uploaded to the backend where proprietary
        AI processing occurs (FAISS search, BM25 ranking, CrossEncoder reranking).
        """)
        
        # API key input
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Required for LLM-powered analysis"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            faiss_file = st.file_uploader(
                "🔍 FAISS Index",
                type=['faiss'],
                help="Pre-computed vector embeddings"
            )
            
            metadata_file = st.file_uploader(
                "📊 Metadata",
                type=['pkl'],
                help="Document chunks and metadata"
            )
        
        with col2:
            bm25_file = st.file_uploader(
                "📈 BM25 Index",
                type=['pkl'],
                help="Keyword search index"
            )
        
        # Upload button
        if all([faiss_file, metadata_file, bm25_file, api_key]):
            if st.button("🚀 Upload to Backend", type="primary", use_container_width=True):
                with st.spinner("⚡ Uploading to secure backend..."):
                    result = upload_files_to_backend(
                        faiss_file,
                        metadata_file,
                        bm25_file,
                        api_key
                    )
                    
                    if result and result.get('success'):
                        st.success(f"✅ Loaded {result.get('chunks_loaded', 0)} document chunks!")
                        
                        # Display stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📚 Chunks", result.get('chunks_loaded', 0))
                        with col2:
                            st.metric("🔍 Vector Dim", result.get('vector_dim', 0))
                        with col3:
                            st.metric("✅ Status", "Ready")
    
    # ========================================================================
    # TAB 2: AI Analysis Chat
    # ========================================================================
    
    with tab2:
        if not st.session_state.files_uploaded:
            st.warning("⚠️ Please upload documents first (Upload tab)")
        else:
            st.markdown("### 💬 AI Financial Analyst")
            
            # Example queries
            example_cols = st.columns(4)
            examples = [
                ("📈 Revenue", "What was total revenue?"),
                ("💰 Margins", "Calculate profit margins"),
                ("🎯 P/E Ratio", "What's the P/E ratio?"),
                ("⚖️ Compare", f"Compare with MSFT")
            ]
            
            for col, (label, query) in zip(example_cols, examples):
                with col:
                    if st.button(label, use_container_width=True):
                        st.session_state.query_input = query
                        st.rerun()
            
            st.markdown("---")
            
            # Chat history display
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 AI Analyst:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show reasoning steps (optional)
                    if message.get("reasoning"):
                        with st.expander("🧠 View AI Reasoning Steps"):
                            reasoning = message["reasoning"]
                            st.json(reasoning)
            
            # Query input
            query = st.text_area(
                "💬 Your Question:",
                value=st.session_state.get('query_input', ''),
                placeholder=f"Ask anything about {selected_company}...",
                height=100
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                send_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.query_input = ""
                    st.rerun()
            
            # Process query
            if send_button and query:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })
                
                # Show loading animation
                thinking_placeholder = st.empty()
                with thinking_placeholder:
                    show_skeleton_loader()
                
                # Call backend API for analysis
                result = analyze_query(
                    query=query,
                    company=selected_company,
                    ticker=company_ticker
                )
                
                thinking_placeholder.empty()
                
                if result:
                    # Track analytics
                    track_query_analytics(
                        query,
                        result.get('tools_used', []),
                        result.get('response_time', 0),
                        result.get('success', False)
                    )
                    
                    # Add AI response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get('answer', 'No response'),
                        "reasoning": {
                            "thoughts": result.get('thoughts', []),
                            "actions": result.get('actions', []),
                            "observations": result.get('observations', [])
                        }
                    })
                    
                    st.rerun()
    
    # ========================================================================
    # TAB 3: Analytics Dashboard
    # ========================================================================
    
    with tab3:
        st.markdown("### 📈 Performance Analytics")
        
        if st.session_state.analytics['total_queries'] == 0:
            st.info("📊 No queries yet. Start analyzing to see statistics!")
        else:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Queries",
                    st.session_state.analytics['total_queries']
                )
            
            with col2:
                avg_time = sum(st.session_state.analytics['response_times']) / len(
                    st.session_state.analytics['response_times']
                )
                st.metric("Avg Response", f"{avg_time:.2f}s")
            
            with col3:
                success_rate = (
                    st.session_state.analytics['successful_queries'] / 
                    st.session_state.analytics['total_queries'] * 100
                )
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col4:
                st.metric(
                    "Backend",
                    "Connected" if st.session_state.backend_connected else "Offline"
                )
            
            st.markdown("---")
            
            # Query history
            st.markdown("#### 📜 Recent Queries")
            
            if st.session_state.analytics['query_history']:
                for item in st.session_state.analytics['query_history'][-5:]:
                    with st.expander(f"🔍 {item['query'][:50]}..."):
                        st.write(f"**Time:** {item['timestamp']}")
                        st.write(f"**Response Time:** {item['time']:.2f}s")
                        st.write(f"**Tools Used:** {', '.join(item['tools'])}")
                        st.write(f"**Status:** {'✅ Success' if item['success'] else '❌ Failed'}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏗️ Built with FastAPI + Streamlit | 🔒 Secure API Architecture</p>
        <p style="font-size: 0.9rem;">AI Processing: FAISS + BM25 Hybrid Search | Groq LPU | ReAct Agent</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
