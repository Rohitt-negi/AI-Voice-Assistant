# 🤖 FORTIVEX AI — Autonomous Sales Voice Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20(Free)-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An AI-powered voice sales assistant that qualifies leads, answers product questions, and books appointments — autonomously.**

*B.Tech Final Year Project | FORTIVEX AI Private Limited*

</div>

---

## 📌 Problem Statement

Sales teams spend 60–70% of their time on repetitive tasks: answering the same product questions, manually qualifying leads, and chasing appointments. This results in high labor costs, missed follow-ups, and inconsistent customer experiences — especially for small and mid-sized businesses that cannot afford large sales teams.

## 💡 Solution

An Autonomous AI Sales Voice Agent that:
- Communicates with customers via voice and text 24/7
- Retrieves accurate product information using RAG
- Qualifies leads automatically with a scoring engine
- Books appointments and stores everything in a database
- Gives sales managers a real-time analytics dashboard

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Voice Input | Record microphone audio → Whisper STT |
| 🔊 Voice Output | AI responses spoken aloud via Edge TTS |
| 🧠 AI Conversations | Groq LLM (Llama 3.3 70B) with memory |
| 📚 RAG Knowledge | FAISS + Sentence Transformers retrieval |
| 🎯 Lead Scoring | Automatic Hot/Warm/Cold classification |
| 😊 Sentiment Analysis | DistilBERT or rule-based fallback |
| 📅 Appointment Booking | Slot selection + SQLite storage |
| 📊 Analytics Dashboard | Plotly charts, funnel, conversion metrics |
| 🔄 Fallback Mode | Works without API key (demo mode) |

---

## 🏗️ Architecture

```
Customer Input (Voice/Text)
        ↓
   Speech-to-Text (Whisper)
        ↓
   RAG Retrieval (FAISS + Sentence Transformers)
        ↓
   AI Agent (Groq LLM - Llama 3.3 70B)
        ↓
   Sentiment Analysis (DistilBERT)
        ↓
   Lead Qualification (Rule Engine)
        ↓
   Database (SQLite)  ←→  TTS (Edge TTS)
        ↓
   Streamlit Dashboard
```

See [architecture.md](architecture.md) for the full Mermaid diagram.

---

## 🛠️ Technology Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.10+ | Ecosystem for AI/ML |
| Frontend | Streamlit | Rapid UI for data apps |
| LLM | Groq API (free tier) | 500 tok/s, Llama 3.3 70B |
| STT | OpenAI Whisper (local) | Free, accurate, offline |
| TTS | Microsoft Edge TTS | Free, natural voices |
| RAG | FAISS + LangChain | Semantic search |
| Embeddings | Sentence Transformers | Free, local, 80MB |
| Sentiment | DistilBERT (HuggingFace) | Free, 250MB, accurate |
| Database | SQLite | Zero-config, file-based |
| Visualization | Plotly | Interactive charts |

---

## 📁 Project Structure

```
autonomous-ai-sales-voice-agent/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
├── README.md
├── PROJECT_REPORT.md
├── architecture.md
│
├── modules/
│   ├── __init__.py
│   ├── ai_agent.py           # LLM conversation engine
│   ├── speech_to_text.py     # Whisper STT
│   ├── text_to_speech.py     # Edge TTS
│   ├── rag.py                # RAG with FAISS
│   ├── lead_qualification.py # Lead scoring engine
│   ├── sentiment.py          # Sentiment analysis
│   ├── appointment.py        # Slot management
│   └── database.py           # SQLite operations
│
├── data/
│   └── sample_company_knowledge.txt   # Company knowledge base
│
└── assets/
    └── styles.css            # Custom CSS theme
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/autonomous-ai-sales-voice-agent.git
cd autonomous-ai-sales-voice-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads Whisper (~140MB) and DistilBERT (~250MB) models automatically.

### 4. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open: **http://localhost:8501**

---

## 🔑 Getting Your Free Groq API Key

1. Go to **https://console.groq.com/**
2. Sign up with your Google or GitHub account (free)
3. Click **"Create API Key"**
4. Copy the key and paste into your `.env` file

**Free tier limits:** 14,400 requests/day, 100 requests/minute — more than enough for demos.

---

## 🎯 Complete Workflow Demo

1. Open the app → Go to **🎙️ Voice Agent**
2. Click **Start Conversation**
3. ARIA greets you and starts listening
4. Ask about products: *"Tell me about VoiceAgent Pro"*
5. ARIA retrieves info from the knowledge base using RAG
6. Ask pricing: *"What are your pricing plans?"*
7. Share your info: *"My name is Rohit, email is rohit@company.com"*
8. ARIA qualifies you as a lead automatically
9. Book a demo: Go to **📅 Book Appointment**
10. View your lead in **👥 Lead Management**
11. End conversation → AI generates call summary
12. View analytics in **📊 Call Analytics**

---

## 🌐 API Fallback Mode

The application works **without a Groq API key** in demo mode:
- Responses are generated using intelligent rule-based logic
- All other features (RAG, lead scoring, appointments, DB) work fully
- Perfect for demonstrating the workflow offline

---

## 📸 Screenshots

| Dashboard | Voice Agent | Lead Management |
|---|---|---|
| *Real-time KPIs and charts* | *Chat interface with voice* | *Lead cards with scoring* |

---

## 🚀 Future Improvements

1. **WebRTC Integration** — Real browser microphone via `streamlit-webrtc`
2. **Telephony** — Twilio/Plivo integration for real phone calls
3. **Multi-language** — Hindi, Tamil, Telugu support
4. **CRM Integration** — HubSpot, Salesforce, Zoho sync
5. **Email Automation** — Automatic follow-up emails post-call
6. **WhatsApp Bot** — WhatsApp Business API integration
7. **Custom Model** — Fine-tune on company-specific conversations
8. **Redis Caching** — Session state management at scale
9. **Docker Deployment** — Containerized production deployment
10. **A/B Testing** — Compare different sales scripts

---

## 📈 Scalability Plan (100,000 Users)

| Component | Current MVP | At Scale |
|---|---|---|
| Database | SQLite (file) | PostgreSQL + pgBouncer |
| LLM | Groq free tier | Groq Enterprise / Self-hosted |
| Session State | Streamlit session | Redis cluster |
| Frontend | Single Streamlit | FastAPI + React |
| Deployment | Local | Kubernetes on GCP/AWS |
| Queue | None | Celery + RabbitMQ |
| Storage | Local filesystem | AWS S3 / GCS |

---

## 🤔 Technical Interview Prep

**Q: How does the RAG system work?**
> The knowledge base text is split into 500-character chunks, embedded using Sentence Transformers (all-MiniLM-L6-v2), and stored in a FAISS vector index. For each user query, the same embedding model encodes the query and FAISS finds the top-3 most similar chunks using cosine similarity. These chunks are injected into the LLM prompt as context.

**Q: How do you prevent duplicate appointment bookings?**
> The SQLite `appointments` table has a `UNIQUE(appointment_date, appointment_time)` constraint. Any duplicate insert raises an `IntegrityError` which is caught and shown to the user.

**Q: Why Groq over OpenAI?**
> Groq offers a free tier with 14,400 requests/day and runs at 500+ tokens/second (10x faster than OpenAI). The Llama 3.3 70B model on Groq matches GPT-4 quality for conversational tasks.

**Q: How does sentiment analysis work?**
> We use DistilBERT fine-tuned on SST-2 (a sentiment dataset). It classifies text as POSITIVE/NEGATIVE with a confidence score. If confidence < 0.6, we classify as Neutral. If the model fails to load, we fall back to keyword-based rules.

---

## 👨‍💻 Author

Built for **FORTIVEX AI Private Limited** technical evaluation.
B.Tech Final Year Project — AI/ML Track

---

## 📄 License

MIT License — Free for educational and personal use.
