"""
=====================================================
FORTIVEX AI - Autonomous Sales Voice Agent
Main Application Entry Point (app.py)

Run with: streamlit run app.py

This is the main Streamlit app that assembles all
pages (Dashboard, Voice Agent, Lead Management,
Appointments, Analytics) and manages shared state.
=====================================================
"""

import os
import sys
import uuid
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- Add modules to path ---
sys.path.insert(0, str(Path(__file__).parent))

# --- Import our modules ---
from modules.database import (
    initialize_database, save_lead, save_message, get_all_leads,
    get_dashboard_stats, get_conversation_history, get_all_conversations,
    book_appointment, get_booked_slots, get_all_appointments,
    save_call_summary, get_call_summaries, get_lead_by_session
)
from modules.ai_agent import SalesAgent
from modules.sentiment import analyze_sentiment, get_sentiment_emoji, get_sentiment_color
from modules.lead_qualification import calculate_lead_score, get_next_action, get_lead_color
from modules.appointment import (
    get_available_slots_only, format_slots_for_display,
    APPOINTMENT_TYPES, get_appointment_confirmation_message, validate_appointment_data
)
from modules.text_to_speech import synthesize_speech, clean_text_for_tts, AVAILABLE_VOICES
from modules.rag import get_rag_instance, initialize_rag

# =============================================================
# PAGE CONFIGURATION (must be first Streamlit call)
# =============================================================
st.set_page_config(
    page_title="FORTIVEX AI Sales Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:support@fortivex.ai",
        "About": "FORTIVEX AI - Autonomous Sales Voice Agent v1.0"
    }
)

# =============================================================
# LOAD CUSTOM CSS
# =============================================================
def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================================================
# INITIALIZE DATABASE AND RAG (once per app start)
# =============================================================
@st.cache_resource
def setup_app():
    """Initialize all backend services once."""
    initialize_database()
    rag = get_rag_instance()
    if not rag._initialized:
        logger.info("Initializing RAG knowledge base...")
        rag.initialize()
    return True

setup_app()

# =============================================================
# SESSION STATE INITIALIZATION
# =============================================================
def init_session_state():
    """Initialize all Streamlit session state variables."""
    defaults = {
        "session_id": str(uuid.uuid4()),
        "agent": None,
        "conversation": [],       # [{role, content, sentiment, timestamp}]
        "customer_info": {},
        "lead_data": {},
        "session_active": False,
        "appointment_booked": False,
        "audio_enabled": True,
        "tts_voice": "en-US-AriaNeural",
        "last_audio_path": None,
        "current_page": "🏠 Dashboard",
        "rag_initialized": False,
        "session_start_time": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def get_or_create_agent() -> SalesAgent:
    """Get the existing AI agent or create a new one."""
    if st.session_state.agent is None:
        st.session_state.agent = SalesAgent()
    return st.session_state.agent


def send_message(user_text: str):
    """
    Process a user message through the full pipeline:
    1. Analyze sentiment
    2. Retrieve RAG context
    3. Get AI response
    4. Update lead score
    5. Save to database
    6. Optionally generate speech
    """
    if not user_text.strip():
        return

    agent = get_or_create_agent()
    session_id = st.session_state.session_id

    # --- Step 1: Analyze customer message sentiment ---
    sentiment_result = analyze_sentiment(user_text)
    sentiment_label = sentiment_result["label"]

    # --- Step 2: RAG context retrieval ---
    rag = get_rag_instance()
    rag_context = rag.get_context_string(user_text) if rag._initialized else ""

    # --- Step 3: Get AI response ---
    with st.spinner("ARIA is thinking..."):
        ai_response = agent.get_response(user_text, rag_context=rag_context)

    # --- Step 4: Store in conversation ---
    timestamp = datetime.now().strftime("%H:%M")

    st.session_state.conversation.append({
        "role": "user",
        "content": user_text,
        "sentiment": sentiment_label,
        "timestamp": timestamp
    })
    st.session_state.conversation.append({
        "role": "assistant",
        "content": ai_response,
        "sentiment": "N/A",
        "timestamp": timestamp
    })

    # --- Step 5: Save to database ---
    save_message(session_id, "user", user_text, sentiment_label)
    save_message(session_id, "assistant", ai_response)

    # --- Step 6: Update lead data ---
    customer_info = agent.customer_info.copy()
    full_conv = " ".join([m["content"] for m in st.session_state.conversation])
    lead_result = calculate_lead_score(customer_info, full_conv)

    lead_data = {
        **customer_info,
        "lead_score": lead_result["score"],
        "lead_classification": lead_result["classification"],
        "classification_reason": lead_result["reason"],
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_result["score"],
        "appointment_booked": st.session_state.appointment_booked
    }
    st.session_state.lead_data = lead_data
    st.session_state.customer_info = customer_info
    save_lead(session_id, lead_data)

    # --- Step 7: Text-to-speech ---
    if st.session_state.audio_enabled:
        try:
            clean_text = clean_text_for_tts(ai_response)
            audio_path = synthesize_speech(
                clean_text,
                voice=st.session_state.tts_voice
            )
            st.session_state.last_audio_path = audio_path
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
            st.session_state.last_audio_path = None


def start_new_session():
    """Reset everything for a fresh conversation."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.agent = SalesAgent()
    st.session_state.conversation = []
    st.session_state.customer_info = {}
    st.session_state.lead_data = {}
    st.session_state.session_active = True
    st.session_state.appointment_booked = False
    st.session_state.last_audio_path = None
    st.session_state.session_start_time = datetime.now()

    # Add greeting message
    agent = get_or_create_agent()
    greeting = agent.generate_greeting()
    st.session_state.conversation.append({
        "role": "assistant",
        "content": greeting,
        "sentiment": "N/A",
        "timestamp": datetime.now().strftime("%H:%M")
    })
    save_message(st.session_state.session_id, "assistant", greeting)

    # TTS for greeting
    if st.session_state.audio_enabled:
        try:
            audio_path = synthesize_speech(
                clean_text_for_tts(greeting),
                voice=st.session_state.tts_voice
            )
            st.session_state.last_audio_path = audio_path
        except Exception:
            pass


# =============================================================
# SIDEBAR NAVIGATION
# =============================================================
def render_sidebar():
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size:2.5rem; margin-bottom:0.3rem;">🤖</div>
            <h2 style="color:#63A2FF; margin:0; font-size:1.1rem; font-weight:700; letter-spacing:0.05em;">
                FORTIVEX AI
            </h2>
            <p style="color:#8BAFD4; font-size:0.75rem; margin:0;">Sales Voice Agent v1.0</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        pages = {
            "🏠 Dashboard": "Dashboard",
            "🎙️ Voice Agent": "Voice Agent",
            "👥 Lead Management": "Lead Management",
            "📅 Book Appointment": "Appointments",
            "📊 Call Analytics": "Analytics"
        }

        st.markdown("**Navigation**")
        selected = st.radio(
            "nav",
            list(pages.keys()),
            label_visibility="collapsed",
            key="nav_radio"
        )
        st.session_state.current_page = selected

        st.divider()

        # Agent Status
        is_active = st.session_state.session_active
        status_class = "status-active" if is_active else "status-idle"
        status_text = "Active Session" if is_active else "No Active Session"
        st.markdown(f"""
        <div style="margin-bottom:1rem;">
            <p style="color:#8BAFD4; font-size:0.75rem; margin:0 0 0.3rem 0; text-transform:uppercase; font-weight:600;">
                AGENT STATUS
            </p>
            <p class="{status_class}" style="color:#E8EDF5; font-size:0.9rem; margin:0;">
                {status_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # TTS Toggle
        st.session_state.audio_enabled = st.toggle(
            "🔊 Voice Output (TTS)",
            value=st.session_state.audio_enabled
        )

        if st.session_state.audio_enabled:
            voice_name = st.selectbox(
                "Voice",
                list(AVAILABLE_VOICES.keys()),
                key="voice_select"
            )
            st.session_state.tts_voice = AVAILABLE_VOICES[voice_name]

        st.divider()

        # Quick stats
        stats = get_dashboard_stats()
        st.markdown("**Quick Stats**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Leads", stats["total_leads"])
            st.metric("Hot 🔥", stats["hot_leads"])
        with col2:
            st.metric("Bookings", stats["appointments_booked"])
            st.metric("Warm 🌡️", stats["warm_leads"])

        st.divider()
        st.markdown(
            "<p style='color:#8BAFD4; font-size:0.7rem; text-align:center;'>"
            "FORTIVEX AI Private Limited<br/>© 2025 All rights reserved</p>",
            unsafe_allow_html=True
        )


# =============================================================
# PAGE: DASHBOARD
# =============================================================
def page_dashboard():
    st.markdown("## 🏠 Sales Dashboard")
    st.markdown("Real-time overview of your AI sales agent performance.")
    st.divider()

    stats = get_dashboard_stats()

    # --- KPI Metrics Row ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Leads", stats["total_leads"], delta=f"+{stats['total_leads']}")
    with col2:
        st.metric("🔥 Hot Leads", stats["hot_leads"])
    with col3:
        st.metric("🌡️ Warm Leads", stats["warm_leads"])
    with col4:
        st.metric("❄️ Cold Leads", stats["cold_leads"])
    with col5:
        st.metric("📅 Appointments", stats["appointments_booked"])

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # --- Lead Classification Chart ---
        st.markdown("### Lead Distribution")
        lead_labels = ["Hot Lead", "Warm Lead", "Cold Lead"]
        lead_values = [stats["hot_leads"], stats["warm_leads"], stats["cold_leads"]]
        lead_colors = ["#E74C3C", "#F39C12", "#3498DB"]

        if sum(lead_values) > 0:
            fig_leads = go.Figure(data=[go.Pie(
                labels=lead_labels,
                values=lead_values,
                hole=0.55,
                marker=dict(colors=lead_colors, line=dict(color="#0A0E1A", width=2)),
                textinfo="label+percent",
                textfont=dict(color="white", size=12)
            )])
            fig_leads.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=280,
                annotations=[dict(
                    text=f"<b>{stats['total_leads']}</b><br>Total",
                    x=0.5, y=0.5, font_size=16, showarrow=False, font_color="white"
                )]
            )
            st.plotly_chart(fig_leads, use_container_width=True)
        else:
            st.info("No lead data yet. Start a conversation to generate leads!")

        # --- Sentiment Distribution ---
        st.markdown("### Sentiment Distribution")
        sentiment_dist = stats.get("sentiment_distribution", {})
        if sentiment_dist:
            sent_labels = list(sentiment_dist.keys())
            sent_values = list(sentiment_dist.values())
            sent_colors = [get_sentiment_color(l) for l in sent_labels]
            fig_sent = go.Figure(data=[go.Bar(
                x=sent_labels,
                y=sent_values,
                marker_color=sent_colors,
                text=sent_values,
                textposition="outside",
                textfont=dict(color="white")
            )])
            fig_sent.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                margin=dict(t=20, b=10, l=10, r=10),
                height=200
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("No sentiment data yet.")

    with col_right:
        # --- Conversion Funnel ---
        st.markdown("### Conversion Funnel")
        total = max(stats["total_leads"], 1)
        funnel_stages = ["Conversations", "Leads Qualified", "Demos Booked"]
        funnel_values = [
            stats["total_conversations"],
            stats["total_leads"],
            stats["appointments_booked"]
        ]
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textinfo="value+percent initial",
            marker=dict(color=["#1A5FFF", "#F39C12", "#2ECC71"]),
            connector=dict(line=dict(color="rgba(99,162,255,0.3)"))
        ))
        fig_funnel.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=250
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        # --- Recent Activity ---
        st.markdown("### Recent Activity")
        recent = stats.get("recent_leads", [])
        if recent:
            for item in recent:
                name = item.get("name") or "Unknown"
                classification = item.get("lead_classification", "Cold Lead")
                sentiment = item.get("sentiment", "Neutral")
                created_at = item.get("created_at", "")

                color = get_lead_color(classification)
                emoji = get_sentiment_emoji(sentiment)

                st.markdown(f"""
                <div style="
                    background: rgba(13,27,53,0.8);
                    border: 1px solid rgba(99,162,255,0.15);
                    border-radius:8px; padding:0.6rem 0.8rem;
                    margin-bottom:0.4rem; display:flex;
                    justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:600; color:#E8EDF5;">{name}</span>
                        <span style="color:{color}; font-size:0.75rem; margin-left:0.5rem;">
                            ● {classification}
                        </span>
                    </div>
                    <span style="font-size:1.1rem;">{emoji}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity yet.")


# =============================================================
# PAGE: VOICE AGENT
# =============================================================
def page_voice_agent():
    st.markdown("## 🎙️ AI Voice Sales Agent")
    st.markdown("Interact with ARIA — your intelligent AI sales representative.")
    st.divider()

    col_chat, col_info = st.columns([3, 2])

    with col_chat:
        # --- Session Controls ---
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if not st.session_state.session_active:
                if st.button("▶️ Start Conversation", use_container_width=True):
                    start_new_session()
                    st.rerun()
            else:
                if st.button("⏹️ End Conversation", use_container_width=True):
                    _end_conversation()
                    st.rerun()

        with ctrl_col2:
            if st.button("🔄 New Session", use_container_width=True):
                start_new_session()
                st.rerun()

        # --- Audio Playback ---
        if st.session_state.last_audio_path and st.session_state.audio_enabled:
            audio_path = st.session_state.last_audio_path
            if os.path.exists(audio_path):
                try:
                    with open(audio_path, "rb") as af:
                        audio_bytes = af.read()
                    ext = os.path.splitext(audio_path)[1].lower().strip(".")
                    mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
                    st.audio(audio_bytes, format=mime, autoplay=True)
                except Exception as e:
                    logger.warning(f"Audio playback error: {e}")

        st.divider()

        # --- Conversation Display ---
        st.markdown("### 💬 Conversation")

        conversation_container = st.container()
        with conversation_container:
            if not st.session_state.conversation:
                st.markdown("""
                <div style="text-align:center; padding:3rem 1rem; color:#8BAFD4;">
                    <div style="font-size:3rem; margin-bottom:1rem;">🤖</div>
                    <h3 style="color:#8BAFD4;">ARIA is ready</h3>
                    <p>Click "Start Conversation" to begin your sales interaction.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.conversation:
                    role = msg["role"]
                    content = msg["content"]
                    sentiment = msg.get("sentiment", "")
                    timestamp = msg.get("timestamp", "")

                    if role == "user":
                        sent_emoji = get_sentiment_emoji(sentiment) if sentiment != "N/A" else ""
                        st.markdown(f"""
                        <div class="chat-message-user">
                            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                                <span style="font-weight:600; color:#63A2FF; font-size:0.8rem;">You {sent_emoji}</span>
                                <span style="color:#8BAFD4; font-size:0.75rem;">{timestamp}</span>
                            </div>
                            <p style="margin:0; color:#E8EDF5;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message-assistant">
                            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                                <span style="font-weight:600; color:#1A5FFF; font-size:0.8rem;">🤖 ARIA</span>
                                <span style="color:#8BAFD4; font-size:0.75rem;">{timestamp}</span>
                            </div>
                            <p style="margin:0; color:#E8EDF5; white-space:pre-wrap;">{content}</p>
                        </div>
                        """, unsafe_allow_html=True)

        st.divider()

        # --- Input Section ---
        if st.session_state.session_active:
            # Text input
            with st.form("message_form", clear_on_submit=True):
                input_col1, input_col2 = st.columns([5, 1])
                with input_col1:
                    user_text = st.text_input(
                        "Message",
                        placeholder="Type your message or ask about our products...",
                        label_visibility="collapsed",
                        key="text_input_field"
                    )
                with input_col2:
                    submitted = st.form_submit_button("Send ➤", use_container_width=True)

                if submitted and user_text.strip():
                    send_message(user_text)
                    st.rerun()

            # Voice recording section
            st.markdown("**🎤 Voice Input**")
            voice_col1, voice_col2 = st.columns(2)
            with voice_col1:
                duration = st.slider("Recording duration (seconds)", 3, 15, 5)
            with voice_col2:
                if st.button("🎤 Record & Transcribe", use_container_width=True):
                    _handle_voice_input(duration)

            # File upload for speech
            st.markdown("**📎 Or upload an audio file:**")
            uploaded_audio = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a", "ogg"],
                label_visibility="collapsed"
            )
            if uploaded_audio:
                _handle_audio_upload(uploaded_audio)

        else:
            st.info("👆 Click **Start Conversation** to interact with ARIA.")

    with col_info:
        # --- Customer Info Panel ---
        st.markdown("### 👤 Customer Profile")

        info = st.session_state.customer_info
        lead = st.session_state.lead_data

        # Completion progress
        filled_fields = sum(1 for v in [
            info.get("name"), info.get("email"), info.get("phone"),
            info.get("product_interest"), info.get("budget"), info.get("timeline")
        ] if v)
        progress = filled_fields / 6
        st.progress(progress, text=f"Profile {int(progress*100)}% complete")

        # Info fields
        fields = [
            ("👤", "Name", info.get("name", "—")),
            ("📧", "Email", info.get("email", "—")),
            ("📱", "Phone", info.get("phone", "—")),
            ("🏢", "Company", info.get("company", "—")),
            ("🛒", "Product Interest", info.get("product_interest", "—")),
            ("💰", "Budget", info.get("budget", "—")),
            ("📆", "Timeline", info.get("timeline", "—")),
        ]

        for icon, label, value in fields:
            color = "#E8EDF5" if value != "—" else "#8BAFD4"
            st.markdown(f"""
            <div style="
                background: rgba(13,27,53,0.6);
                border: 1px solid rgba(99,162,255,0.1);
                border-radius:8px; padding:0.5rem 0.75rem;
                margin-bottom:0.3rem;">
                <span style="color:#8BAFD4; font-size:0.75rem;">{icon} {label}</span><br>
                <span style="color:{color}; font-weight:500;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- Lead Score Panel ---
        st.markdown("### 📈 Lead Score")
        if lead:
            score = lead.get("lead_score", 0)
            classification = lead.get("lead_classification", "Cold Lead")
            reason = lead.get("classification_reason", "")
            color = get_lead_color(classification)

            # Score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": color},
                    "bgcolor": "#0D1B35",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(52,152,219,0.1)"},
                        {"range": [40, 70], "color": "rgba(243,156,18,0.1)"},
                        {"range": [70, 100], "color": "rgba(231,76,60,0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 3},
                        "thickness": 0.75,
                        "value": score
                    }
                },
                number={"suffix": "/100", "font": {"color": "white", "size": 24}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(t=0, b=0, l=10, r=10),
                height=180
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div style="
                background: rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))}, 0.1);
                border: 1px solid {color};
                border-radius:8px; padding:0.6rem;
                text-align:center; margin-bottom:0.5rem;">
                <span style="color:{color}; font-weight:700; font-size:1rem;">
                    {classification}
                </span>
            </div>
            """, unsafe_allow_html=True)

            if reason:
                st.markdown(f"<small style='color:#8BAFD4;'>{reason}</small>", unsafe_allow_html=True)

            # Sentiment
            st.divider()
            st.markdown("### 😊 Sentiment")
            sentiment = lead.get("sentiment", "Neutral")
            sent_emoji = get_sentiment_emoji(sentiment)
            sent_color = get_sentiment_color(sentiment)
            st.markdown(f"""
            <div style="text-align:center; padding:0.5rem;">
                <span style="font-size:2rem;">{sent_emoji}</span><br>
                <span style="color:{sent_color}; font-weight:600;">{sentiment}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Lead data will appear after conversation starts.")


def _handle_voice_input(duration: int):
    """Handle voice recording and transcription."""
    from modules.speech_to_text import record_and_transcribe, is_microphone_available

    if not is_microphone_available():
        st.error("❌ No microphone detected. Please use text input instead.")
        return

    with st.spinner(f"🎤 Recording for {duration} seconds..."):
        transcribed = record_and_transcribe(duration_seconds=duration)

    if transcribed:
        st.success(f"Transcribed: *\"{transcribed}\"*")
        send_message(transcribed)
        st.rerun()
    else:
        st.error("Could not transcribe audio. Please try again or use text input.")


def _handle_audio_upload(uploaded_file):
    """Handle uploaded audio file for transcription."""
    from modules.speech_to_text import transcribe_audio_file

    with tempfile.NamedTemporaryFile(
        suffix=f".{uploaded_file.name.split('.')[-1]}",
        delete=False
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing uploaded audio..."):
        transcribed = transcribe_audio_file(tmp_path)

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if transcribed:
        st.success(f"Transcribed: *\"{transcribed}\"*")
        if st.button("Send this message"):
            send_message(transcribed)
            st.rerun()
    else:
        st.error("Transcription failed. Please try another audio file.")


def _end_conversation():
    """End the current conversation and generate call summary."""
    agent = get_or_create_agent()
    session_id = st.session_state.session_id

    if st.session_state.conversation:
        # Generate and save call summary
        summary = agent.generate_call_summary(
            [{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation]
        )

        lead = st.session_state.lead_data
        summary_data = {
            **summary,
            "lead_classification": lead.get("lead_classification", "Cold Lead"),
            "appointment_status": "Booked" if st.session_state.appointment_booked else "Not booked",
            "total_messages": len(st.session_state.conversation),
            "duration_seconds": agent.get_session_duration()
        }
        save_call_summary(session_id, summary_data)

    st.session_state.session_active = False
    st.success("✅ Conversation ended and summary saved!")


# =============================================================
# PAGE: LEAD MANAGEMENT
# =============================================================
def page_lead_management():
    st.markdown("## 👥 Lead Management")
    st.markdown("View and manage all qualified leads from your AI sales conversations.")
    st.divider()

    leads = get_all_leads()

    if not leads:
        st.info("No leads yet. Start a conversation with ARIA to generate your first lead!")
        return

    # --- Filters ---
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        classification_filter = st.selectbox(
            "Filter by Classification",
            ["All", "Hot Lead", "Warm Lead", "Cold Lead"]
        )
    with filter_col2:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Neutral", "Negative"]
        )
    with filter_col3:
        search_term = st.text_input("Search by name or email", placeholder="Search...")

    # --- Apply filters ---
    df = pd.DataFrame(leads)
    if classification_filter != "All":
        df = df[df["lead_classification"] == classification_filter]
    if sentiment_filter != "All":
        df = df[df["sentiment"] == sentiment_filter]
    if search_term:
        mask = (
            df["name"].fillna("").str.contains(search_term, case=False) |
            df["email"].fillna("").str.contains(search_term, case=False)
        )
        df = df[mask]

    st.markdown(f"**Showing {len(df)} lead(s)**")

    # --- Lead Cards ---
    for _, lead in df.iterrows():
        name = lead.get("name") or "Unknown Customer"
        email = lead.get("email") or "—"
        phone = lead.get("phone") or "—"
        company = lead.get("company") or "—"
        classification = lead.get("lead_classification", "Cold Lead")
        score = lead.get("lead_score", 0)
        sentiment = lead.get("sentiment", "Neutral")
        product = lead.get("product_interest") or "Not specified"
        budget = lead.get("budget") or "Not disclosed"
        timeline = lead.get("timeline") or "Not specified"
        reason = lead.get("classification_reason") or ""
        appt = "✅ Booked" if lead.get("appointment_booked") else "❌ Not booked"
        created = str(lead.get("created_at", ""))[:16]

        lead_color = get_lead_color(classification)
        sent_emoji = get_sentiment_emoji(sentiment)
        next_act = get_next_action(classification, bool(lead.get("appointment_booked")))

        with st.expander(f"{sent_emoji} **{name}** | {classification} | Score: {score}/100 | {created}"):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"📧 **Email:** {email}")
                st.markdown(f"📱 **Phone:** {phone}")
                st.markdown(f"🏢 **Company:** {company}")
            with d2:
                st.markdown(f"🛒 **Product:** {product}")
                st.markdown(f"💰 **Budget:** {budget}")
                st.markdown(f"📆 **Timeline:** {timeline}")
            with d3:
                st.markdown(f"📅 **Appointment:** {appt}")
                st.markdown(f"😊 **Sentiment:** {sentiment} {sent_emoji}")
                st.markdown(f"🎯 **Score:** {score}/100")

            if reason:
                st.markdown(f"**Classification:** {reason}")
            st.markdown(f"**Next Action:** {next_act}")


# =============================================================
# PAGE: APPOINTMENTS
# =============================================================
def page_appointments():
    st.markdown("## 📅 Appointment Booking")
    st.markdown("Schedule product demos and sales consultations.")
    st.divider()

    tab1, tab2 = st.tabs(["📋 Book Appointment", "📂 All Appointments"])

    with tab1:
        col_form, col_slots = st.columns([2, 3])

        with col_form:
            st.markdown("### Your Information")
            with st.form("appointment_form"):
                # Pre-fill from customer info if available
                info = st.session_state.customer_info

                cust_name = st.text_input(
                    "Full Name *",
                    value=info.get("name", ""),
                    placeholder="Enter your full name"
                )
                cust_email = st.text_input(
                    "Email Address *",
                    value=info.get("email", ""),
                    placeholder="your@email.com"
                )
                cust_phone = st.text_input(
                    "Phone Number",
                    value=info.get("phone", ""),
                    placeholder="+91 98765 43210"
                )
                appt_type = st.selectbox("Appointment Type *", APPOINTMENT_TYPES)
                notes = st.text_area(
                    "Special Requests (optional)",
                    placeholder="Any specific topics or requirements...",
                    height=80
                )

                st.markdown("*Select a slot from the right panel*")
                book_btn = st.form_submit_button("📅 Book Appointment", use_container_width=True)

        with col_slots:
            st.markdown("### Available Time Slots")
            st.markdown("*All times are in IST (Indian Standard Time)*")

            # Date and time selection
            available_slots = get_available_slots_only(days_ahead=7)

            if not available_slots:
                st.warning("No available slots in the next 7 days. Contact us at sales@fortivex.ai")
            else:
                grouped = format_slots_for_display(available_slots)

                selected_date = st.selectbox(
                    "Select Date",
                    list(grouped.keys()),
                    key="appt_date"
                )

                if selected_date and selected_date in grouped:
                    day_slots = grouped[selected_date]
                    time_options = {s["time_label"]: s for s in day_slots}

                    selected_time_label = st.selectbox(
                        "Select Time",
                        list(time_options.keys()),
                        key="appt_time"
                    )

                    if selected_time_label:
                        selected_slot = time_options[selected_time_label]

                        st.markdown(f"""
                        <div style="
                            background: rgba(26,95,255,0.1);
                            border: 1px solid rgba(26,95,255,0.4);
                            border-radius:10px; padding:1rem;
                            margin:1rem 0;">
                            <h4 style="color:#63A2FF; margin:0 0 0.5rem 0;">Selected Slot</h4>
                            <p style="margin:0.2rem 0; color:#E8EDF5;">
                                📅 <b>{selected_date}</b>
                            </p>
                            <p style="margin:0.2rem 0; color:#E8EDF5;">
                                ⏰ <b>{selected_time_label} IST</b>
                            </p>
                            <p style="margin:0.2rem 0; color:#8BAFD4;">
                                📹 Google Meet (link via email)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Process booking form submission
                        if book_btn:
                            appt_data = {
                                "session_id": st.session_state.session_id,
                                "customer_name": cust_name,
                                "customer_email": cust_email,
                                "customer_phone": cust_phone,
                                "date": selected_slot["date"],
                                "time": selected_slot["time"],
                                "type": appt_type,
                                "notes": notes
                            }
                            errors = validate_appointment_data(appt_data)
                            if errors:
                                for field, msg in errors.items():
                                    st.error(f"⚠️ {msg}")
                            else:
                                result = book_appointment(appt_data)
                                if result["success"]:
                                    st.session_state.appointment_booked = True
                                    # Update lead data
                                    if st.session_state.lead_data:
                                        st.session_state.lead_data["appointment_booked"] = True
                                        save_lead(st.session_state.session_id, st.session_state.lead_data)
                                    # Show confirmation
                                    confirmation = get_appointment_confirmation_message(appt_data)
                                    st.success(confirmation)
                                    st.balloons()
                                else:
                                    st.error(result["message"])

    with tab2:
        st.markdown("### All Booked Appointments")
        appointments = get_all_appointments()

        if not appointments:
            st.info("No appointments booked yet.")
        else:
            df_appt = pd.DataFrame(appointments)
            # Format display columns
            display_cols = {
                "customer_name": "Customer",
                "customer_email": "Email",
                "appointment_date": "Date",
                "appointment_time": "Time",
                "appointment_type": "Type",
                "status": "Status",
                "created_at": "Booked At"
            }
            available_cols = [c for c in display_cols.keys() if c in df_appt.columns]
            df_display = df_appt[available_cols].rename(columns=display_cols)
            st.dataframe(df_display, use_container_width=True, hide_index=True)


# =============================================================
# PAGE: ANALYTICS
# =============================================================
def page_analytics():
    st.markdown("## 📊 Call Analytics")
    st.markdown("Detailed insights into conversations, leads and sentiment.")
    st.divider()

    tab1, tab2 = st.tabs(["📋 Call Summaries", "🗣️ Conversation History"])

    with tab1:
        summaries = get_call_summaries()
        if not summaries:
            st.info("No call summaries yet. End a conversation to generate a summary.")
            return

        for s in summaries:
            customer = s.get("customer_name") or "Unknown"
            classification = s.get("lead_classification", "Cold Lead")
            sentiment = s.get("sentiment_overall", "Neutral")
            created = str(s.get("created_at", ""))[:16]

            with st.expander(f"📞 {customer} | {classification} | {sentiment} | {created}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**📝 Summary:**\n{s.get('summary', 'N/A')}")
                    st.markdown(f"**🛒 Requirements:**\n{s.get('customer_requirements', 'N/A')}")
                with c2:
                    st.markdown(f"**📧 Email:** {s.get('customer_email', '—')}")
                    st.markdown(f"**📅 Appointment:** {s.get('appointment_status', 'Not booked')}")
                    st.markdown(f"**💬 Messages:** {s.get('total_messages', 0)}")
                st.markdown(f"**🎯 Next Action:** {s.get('next_action', 'Follow up')}")

    with tab2:
        conversations = get_all_conversations()
        if not conversations:
            st.info("No conversations recorded yet.")
            return

        for conv in conversations:
            session_id = conv.get("session_id")
            customer = conv.get("customer_name") or "Unknown"
            classification = conv.get("lead_classification", "Cold Lead")
            msgs = conv.get("message_count", 0)
            last_msg = str(conv.get("last_message", ""))[:16]

            with st.expander(f"💬 {customer} | {classification} | {msgs} messages | {last_msg}"):
                history = get_conversation_history(session_id)
                for msg in history:
                    role = msg["role"]
                    content = msg["content"]
                    ts = str(msg.get("timestamp", ""))[:16]
                    sent = msg.get("sentiment", "")

                    if role == "user":
                        sent_e = get_sentiment_emoji(sent)
                        st.markdown(f"**You** {sent_e} `{ts}`")
                        st.markdown(f"> {content}")
                    else:
                        st.markdown(f"**🤖 ARIA** `{ts}`")
                        st.markdown(f"> {content}")
                    st.markdown("---")


# =============================================================
# MAIN APP ENTRY POINT
# =============================================================
def main():
    render_sidebar()

    page = st.session_state.current_page

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🎙️ Voice Agent":
        page_voice_agent()
    elif page == "👥 Lead Management":
        page_lead_management()
    elif page == "📅 Book Appointment":
        page_appointments()
    elif page == "📊 Call Analytics":
        page_analytics()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
