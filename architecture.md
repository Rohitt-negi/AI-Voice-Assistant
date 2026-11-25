# System Architecture — Autonomous AI Sales Voice Agent

## Complete System Architecture

```mermaid
flowchart TD
    subgraph INPUT["🎤 Input Layer"]
        MIC[Microphone\nAudio Input]
        TEXT[Text Input\nKeyboard]
        UPLOAD[Audio File\nUpload]
    end

    subgraph STT["🔊 Speech Processing"]
        WHISPER[Whisper STT\nLocal Model\nbase ~140MB]
        AUDIO_PROC[Audio Processor\nsounddevice + soundfile]
    end

    subgraph RAG_LAYER["📚 Knowledge Retrieval - RAG"]
        KB[Knowledge Base\nTXT / PDF Documents]
        CHUNKER[Text Chunker\n500-char chunks\n50-char overlap]
        EMBEDDER[Sentence Transformers\nall-MiniLM-L6-v2\n80MB Local Model]
        FAISS_IDX[FAISS Vector Index\nCosine Similarity Search]
        RETRIEVER[Top-K Retriever\nK=3 chunks]
    end

    subgraph AI_CORE["🧠 AI Agent Core"]
        SYSTEM_PROMPT[System Prompt\nARIA Persona\nSales Guidelines]
        CONV_HIST[Conversation Memory\nLast 10 Messages]
        GROQ[Groq LLM API\nLlama 3.3 70B\nFree Tier]
        FALLBACK[Rule-Based\nFallback Responses]
        INFO_EXTRACT[Customer Info\nExtractor\nRegex Patterns]
    end

    subgraph ANALYSIS["📊 Analysis Layer"]
        SENTIMENT[Sentiment Analysis\nDistilBERT SST-2\nFree HuggingFace]
        LEAD_QUAL[Lead Qualification\nScoring Engine\n0-100 Points]
        CLASSIFIER[Lead Classifier\nHot / Warm / Cold]
    end

    subgraph OUTPUT["🔊 Output Layer"]
        EDGE_TTS[Edge TTS\nMicrosoft Neural\nFree - No API Key]
        PYTTSX3[pyttsx3\nOffline Fallback TTS]
        AUDIO_OUT[Audio Playback\nStreamlit Audio]
        TEXT_OUT[Text Display\nChat Interface]
    end

    subgraph DB["🗄️ Database - SQLite"]
        LEADS_TBL[(leads)]
        CONV_TBL[(conversations)]
        APPT_TBL[(appointments)]
        SUMMARY_TBL[(call_summaries)]
    end

    subgraph UI["🖥️ Streamlit Frontend"]
        DASHBOARD[Dashboard\nKPIs + Charts]
        VOICE_PAGE[Voice Agent\nChat Interface]
        LEAD_PAGE[Lead Management\nSearch + Filter]
        APPT_PAGE[Book Appointment\nSlot Selection]
        ANALYTICS[Call Analytics\nSummaries]
    end

    %% Input flow
    MIC --> AUDIO_PROC
    UPLOAD --> AUDIO_PROC
    AUDIO_PROC --> WHISPER
    WHISPER --> AI_CORE
    TEXT --> AI_CORE

    %% RAG flow
    KB --> CHUNKER
    CHUNKER --> EMBEDDER
    EMBEDDER --> FAISS_IDX
    AI_CORE --> EMBEDDER
    FAISS_IDX --> RETRIEVER
    RETRIEVER --> AI_CORE

    %% AI processing
    SYSTEM_PROMPT --> GROQ
    CONV_HIST --> GROQ
    AI_CORE --> GROQ
    GROQ -->|"API unavailable"| FALLBACK
    GROQ --> INFO_EXTRACT
    FALLBACK --> INFO_EXTRACT

    %% Analysis
    AI_CORE --> SENTIMENT
    INFO_EXTRACT --> LEAD_QUAL
    SENTIMENT --> LEAD_QUAL
    LEAD_QUAL --> CLASSIFIER

    %% Output
    GROQ --> EDGE_TTS
    FALLBACK --> EDGE_TTS
    EDGE_TTS -->|"TTS fails"| PYTTSX3
    EDGE_TTS --> AUDIO_OUT
    PYTTSX3 --> AUDIO_OUT
    GROQ --> TEXT_OUT
    FALLBACK --> TEXT_OUT

    %% Database writes
    INFO_EXTRACT --> LEADS_TBL
    GROQ --> CONV_TBL
    CLASSIFIER --> LEADS_TBL
    SENTIMENT --> LEADS_TBL
    UI --> APPT_TBL
    AI_CORE --> SUMMARY_TBL

    %% UI reads
    LEADS_TBL --> DASHBOARD
    LEADS_TBL --> LEAD_PAGE
    CONV_TBL --> ANALYTICS
    APPT_TBL --> APPT_PAGE
    SUMMARY_TBL --> ANALYTICS
    TEXT_OUT --> VOICE_PAGE
    AUDIO_OUT --> VOICE_PAGE

    %% Styling
    classDef inputStyle fill:#1A3A6B,stroke:#63A2FF,color:#E8EDF5
    classDef aiStyle fill:#1A1A4B,stroke:#7B68EE,color:#E8EDF5
    classDef ragStyle fill:#1A3A2A,stroke:#2ECC71,color:#E8EDF5
    classDef dbStyle fill:#2A1A1A,stroke:#E74C3C,color:#E8EDF5
    classDef uiStyle fill:#1A2A3A,stroke:#F39C12,color:#E8EDF5
    classDef outputStyle fill:#1A3A3A,stroke:#1ABC9C,color:#E8EDF5

    class MIC,TEXT,UPLOAD inputStyle
    class GROQ,FALLBACK,SYSTEM_PROMPT,CONV_HIST,INFO_EXTRACT aiStyle
    class KB,CHUNKER,EMBEDDER,FAISS_IDX,RETRIEVER ragStyle
    class LEADS_TBL,CONV_TBL,APPT_TBL,SUMMARY_TBL dbStyle
    class DASHBOARD,VOICE_PAGE,LEAD_PAGE,APPT_PAGE,ANALYTICS uiStyle
    class EDGE_TTS,PYTTSX3,AUDIO_OUT,TEXT_OUT outputStyle
```

---

## Module Interaction Flow

```mermaid
sequenceDiagram
    participant U as 👤 Customer
    participant UI as 🖥️ Streamlit UI
    participant STT as 🎤 Whisper STT
    participant RAG as 📚 RAG Engine
    participant AI as 🧠 Groq LLM
    participant SA as 📊 Sentiment
    participant LQ as 🎯 Lead Qualifier
    participant DB as 🗄️ SQLite
    participant TTS as 🔊 Edge TTS

    U->>UI: Speaks / Types message
    UI->>STT: Audio bytes (if voice)
    STT-->>UI: Transcribed text
    UI->>RAG: Query text
    RAG-->>UI: Top-3 relevant chunks
    UI->>AI: Message + RAG context + history
    AI-->>UI: AI response text
    UI->>SA: Customer message
    SA-->>UI: Sentiment label + score
    UI->>LQ: Customer info + conversation
    LQ-->>UI: Lead score + classification
    UI->>DB: Save message, lead, sentiment
    UI->>TTS: AI response text
    TTS-->>UI: Audio file (MP3)
    UI->>U: Display text + play audio
```

---

## Lead Scoring System

```mermaid
flowchart LR
    subgraph INPUTS["Data Points Collected"]
        N[Name\n+10 pts]
        E[Email\n+15 pts]
        P[Phone\n+15 pts]
        C[Company\n+10 pts]
        PI[Product Interest\n+15 pts]
        B[Budget\n+10 pts]
        T[Timeline\n+10 pts]
        R[Requirements\n+5 pts]
    end

    subgraph BONUS["Bonus Scoring"]
        TI[Immediate Timeline\n+15 pts]
        TS[Short Timeline\n+10 pts]
        BH[High Budget ≥50K\n+10 pts]
        BM[Medium Budget\n+5 pts]
        DM[Wants Demo\n+10 pts]
        MP[Multiple Products\n+5 pts]
    end

    subgraph CLASSIFY["Classification"]
        HOT["🔥 Hot Lead\n70-100 pts\nCall within 2 hours"]
        WARM["🌡️ Warm Lead\n40-69 pts\nFollow up in 24 hrs"]
        COLD["❄️ Cold Lead\n0-39 pts\nNurture sequence"]
    end

    INPUTS --> SCORE[Total Score\n0-100]
    BONUS --> SCORE
    SCORE -->|"≥70"| HOT
    SCORE -->|"40-69"| WARM
    SCORE -->|"<40"| COLD
```

---

## Database Schema

```mermaid
erDiagram
    LEADS {
        int id PK
        text session_id UK
        text name
        text email
        text phone
        text company
        text product_interest
        text budget
        text timeline
        text requirements
        int lead_score
        text lead_classification
        text classification_reason
        text sentiment
        real sentiment_score
        int appointment_booked
        timestamp created_at
        timestamp updated_at
    }

    CONVERSATIONS {
        int id PK
        text session_id FK
        text role
        text content
        text sentiment
        timestamp timestamp
    }

    APPOINTMENTS {
        int id PK
        text session_id FK
        text customer_name
        text customer_email
        text customer_phone
        text appointment_date
        text appointment_time
        text appointment_type
        text status
        text notes
        timestamp created_at
    }

    CALL_SUMMARIES {
        int id PK
        text session_id UK FK
        text summary
        text customer_requirements
        text lead_classification
        text sentiment_overall
        text appointment_status
        text next_action
        int total_messages
        int duration_seconds
        timestamp created_at
    }

    LEADS ||--o{ CONVERSATIONS : "has"
    LEADS ||--o| APPOINTMENTS : "books"
    LEADS ||--o| CALL_SUMMARIES : "generates"
```
