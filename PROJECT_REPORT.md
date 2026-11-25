# Project Report: Autonomous AI Sales Voice Agent

**Submitted to:** FORTIVEX AI Private Limited
**Project Type:** B.Tech Final Year Project
**Domain:** Artificial Intelligence, Natural Language Processing, Voice Technology

---

## Project Overview

This project presents an **Autonomous AI Sales Voice Agent** — a fully functional software system that replicates the role of a human sales representative. The agent communicates with customers through voice and text, retrieves product information from a knowledge base, qualifies leads, collects contact information, and books appointments — all without human intervention.

---

## Problem Statement

Modern sales teams face three critical inefficiencies:

1. **Repetitive inquiries** — 70% of customer questions are identical (pricing, features, availability), yet each requires a human to answer
2. **Inconsistent qualification** — lead scoring varies by agent skill and mood, causing missed opportunities
3. **24/7 coverage gap** — businesses lose leads outside business hours

---

## Objectives

- Build a voice-capable AI sales assistant using open-source and free-tier technologies
- Implement RAG-based knowledge retrieval to ensure factually accurate responses
- Automate lead qualification with a transparent, rule-based scoring engine
- Provide a professional, real-time analytics dashboard for sales managers
- Demonstrate a complete end-to-end sales workflow from greeting to appointment booking

---

## Implementation

The system follows a **modular, layered architecture** with 8 independent Python modules:

| Module                  | Responsibility                                                     |
| ----------------------- | ------------------------------------------------------------------ |
| `ai_agent.py`           | Conversation management, LLM integration, customer info extraction |
| `rag.py`                | FAISS vector index, knowledge retrieval, document loading          |
| `sentiment.py`          | DistilBERT-based emotion analysis                                  |
| `lead_qualification.py` | Rule-based scoring (0-100), Hot/Warm/Cold classification           |
| `appointment.py`        | Slot generation, availability, booking validation                  |
| `speech_to_text.py`     | Whisper-based audio transcription                                  |
| `text_to_speech.py`     | Edge TTS voice synthesis with fallback                             |
| `database.py`           | SQLite CRUD for leads, conversations, appointments                 |

---

## Technology Stack

| Layer      | Technology               | Cost                     |
| ---------- | ------------------------ | ------------------------ |
| LLM        | Groq API — Llama 3.3 70B | Free (14,400 req/day)    |
| STT        | OpenAI Whisper (local)   | Free (runs on CPU)       |
| TTS        | Microsoft Edge TTS       | Free (browser interface) |
| Embeddings | all-MiniLM-L6-v2         | Free (80MB, local)       |
| Vector DB  | FAISS (CPU)              | Free                     |
| Sentiment  | DistilBERT SST-2         | Free (250MB, local)      |
| Database   | SQLite 3                 | Free (built-in)          |
| Frontend   | Streamlit                | Free (open-source)       |

**Total infrastructure cost: ₹0/month for the MVP.**

---

## Business Impact

| Metric                  | Traditional Sales Team    | This AI Agent                   |
| ----------------------- | ------------------------- | ------------------------------- |
| Availability            | 9AM-6PM (9 hrs)           | 24/7 (24 hrs)                   |
| Cost per conversation   | ₹150-500 (agent salary)   | Near ₹0                         |
| Lead qualification time | 15-30 minutes             | Instantaneous                   |
| Consistency             | Variable                  | 100% consistent                 |
| Scalability             | Linear (hire more agents) | Instant (deploy more instances) |

---

## Scalability

The current MVP is a single-user Streamlit application. For 100,000 concurrent users, the architecture would evolve to:

- **Backend:** FastAPI with async endpoints replacing Streamlit
- **Database:** PostgreSQL with connection pooling (pgBouncer)
- **Session State:** Redis cluster for distributed session management
- **LLM:** Groq Enterprise or self-hosted Llama via Ollama on GPU servers
- **Queue:** Celery + RabbitMQ for async voice processing jobs
- **Deployment:** Kubernetes (GCP GKE or AWS EKS) with auto-scaling
- **CDN:** Cloudflare for static assets and audio file delivery

This architecture supports horizontal scaling — adding compute nodes without code changes.

---

## Limitations

1. **Whisper requires local compute** — slow on CPU for audio >30 seconds; production would use a GPU server or Whisper API
2. **SQLite is single-process** — suitable for demo; production requires PostgreSQL
3. **Edge TTS requires internet** — offline fallback (pyttsx3) has lower voice quality
4. **No real-time WebRTC** — microphone recording uses sounddevice, not browser WebRTC
5. **Context window limit** — conversation history capped at last 10 messages for Groq token limits
6. **Model cold-start** — DistilBERT and Sentence Transformers take 10-20 seconds to load on first run

---

## Conclusion

This project successfully demonstrates a production-grade AI sales agent workflow using exclusively free and open-source technologies. The modular architecture ensures that any component (LLM, STT, TTS, database) can be swapped independently without affecting the rest of the system. The complete workflow — from customer greeting through lead qualification to appointment booking — functions as a genuine business tool, not a prototype simulation.

---

_FORTIVEX AI Private Limited | Bengaluru, Karnataka | 2025_
