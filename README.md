# 🏦 OmniFin RAG - AI-Powered Financial Analysis Platform

> A production-grade financial document analysis system showcasing modern full-stack architecture, AI engineering, and clean code practices.

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)]()

## 🎯 Overview

OmniFin RAG is an enterprise-grade AI platform for analyzing SEC 10-K filings. This repository showcases the **public-facing architecture** while the proprietary AI components remain secure in a private repository.

### Key Features

- 🤖 **Agentic AI**: ReAct (Reason + Act) pattern for multi-step reasoning
- 🔍 **Hybrid Search**: FAISS vector search + BM25 keyword ranking
- 💬 **Natural Language Interface**: ChatGPT-style conversation UI
- 📊 **Live Market Data**: Real-time stock integration
- 🎨 **Modern UI**: Glassmorphism design with Tailwind-inspired styling
- 🔒 **Secure Architecture**: Client-server separation with API authentication

## 🏗️ Architecture
```
┌─────────────────┐      REST API       ┌──────────────────┐
│                 │ ←─────────────────→ │                  │
│  Streamlit UI   │   JSON over HTTP    │  FastAPI Backend │
│  (This Repo)    │                     │  (Private Repo)  │
│                 │                     │                  │
└─────────────────┘                     └──────────────────┘
        ↓                                         ↓
    User Input                            ┌──────────────────┐
    File Uploads                          │  Proprietary AI  │
    Analytics                             │   - FAISS Search │
                                          │   - ReAct Agent  │
                                          │   - Tool System  │
                                          │   - LLM (Groq)   │
                                          └──────────────────┘
```

### Tech Stack

**Frontend:**
- Streamlit 1.31 - Interactive web UI
- Plotly - Data visualization
- Custom CSS - Glassmorphism design

**Backend API:**
- FastAPI 0.109 - Modern async framework
- Pydantic - Type validation
- Uvicorn - ASGI server

**Proprietary Components (Private):**
- FAISS - Vector similarity search
- SentenceTransformers - Embeddings
- Groq LPU - LLM inference
- Custom ReAct Agent - Multi-step reasoning

## 📁 Repository Structure
```
omnifin-rag/
├── frontend/
│   └── app.py              # Streamlit UI (public)
├── backend/
│   └── main.py             # FastAPI stubs (public blueprint)
├── requirements.txt        # Public dependencies only
├── .gitignore             # Protects proprietary files
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/omnifin-rag.git
cd omnifin-rag

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Note:** This public repository contains the UI and API blueprint only. To run the full application, you need access to the private backend implementation.
```bash
# Frontend only (UI demonstration)
cd frontend
streamlit run app.py

# Backend stubs (API documentation)
cd backend
uvicorn main:app --reload
```

Visit:
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

## 🎨 UI Showcase

The frontend demonstrates:
- ✨ Animated glassmorphism effects
- 💬 ChatGPT-style conversation interface
- 📊 Real-time analytics dashboard
- 📱 Mobile-responsive design
- 🎯 Skeleton loading states
- 🎨 Custom gradient themes

## 🔒 Security & IP Protection

This public repository **deliberately excludes**:
- ❌ FAISS implementation
- ❌ ReAct agent logic
- ❌ System prompts
- ❌ Embedding models
- ❌ Training data
- ❌ Vector indices
- ❌ API keys

All proprietary components are maintained in a separate, private repository.

## 📚 API Documentation

The `/backend/main.py` file provides detailed documentation of the API design, including:

- **POST /upload**: Document processing endpoint
- **POST /analyze**: AI query analysis endpoint
- **GET /health**: Service health check

Each endpoint includes comprehensive docstrings explaining the (private) implementation architecture.

## 🎓 Educational Purpose

This repository serves as a **portfolio piece** demonstrating:

1. **Clean Code**: PEP 8 compliance, type hints, comprehensive docstrings
2. **System Design**: Client-server architecture, API design, separation of concerns
3. **Modern Stack**: FastAPI, Streamlit, Pydantic
4. **Security**: API authentication, input validation, secret management
5. **UX Design**: Professional UI, responsive layout, accessibility

## 📄 License

**Proprietary** - All rights reserved. This code is provided for demonstration and portfolio purposes only.

## 👤 Author

Poorna Tejasvi HV
- Final Year Project - B.E Artifical Intelligence
- AI/ML Engineering Focus
- Full-Stack Development

## 🤝 For Recruiters

This repository showcases my ability to:
- ✅ Design scalable microservices architectures
- ✅ Build production-grade APIs with FastAPI
- ✅ Create professional user interfaces
- ✅ Implement proper security practices
- ✅ Write clean, documented, maintainable code
- ✅ Balance open-source contribution with IP protection

For access to the full implementation including proprietary AI components, please contact me directly.

---

**Note:** The actual AI processing logic (FAISS search, ReAct agent, prompt engineering) is proprietary and maintained in a private repository. This public version demonstrates the architecture and interface design.
