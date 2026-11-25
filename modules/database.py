"""
=====================================================
Module: database.py
Purpose: All SQLite database operations for the
         Sales Voice Agent application.
         Creates tables, saves leads, conversations,
         appointments and retrieves analytics data.
=====================================================
"""

import sqlite3
import os
import json
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database path from environment or default
DB_PATH = os.getenv("DATABASE_PATH", "data/sales_agent.db")


@contextmanager
def get_connection():
    """
    Context manager for database connections.
    Ensures connection is always closed properly,
    even if an exception occurs.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def initialize_database():
    """
    Creates all required database tables if they do not exist.
    Call this once at application startup.
    """
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)

    with get_connection() as conn:
        cursor = conn.cursor()

        # --- Leads Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                name TEXT,
                email TEXT,
                phone TEXT,
                company TEXT,
                product_interest TEXT,
                budget TEXT,
                timeline TEXT,
                requirements TEXT,
                lead_score INTEGER DEFAULT 0,
                lead_classification TEXT DEFAULT 'Cold Lead',
                classification_reason TEXT,
                sentiment TEXT DEFAULT 'Neutral',
                sentiment_score REAL DEFAULT 0.0,
                appointment_booked INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- Conversations Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sentiment TEXT DEFAULT 'Neutral',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES leads(session_id)
            )
        """)

        # --- Appointments Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                customer_name TEXT,
                customer_email TEXT,
                customer_phone TEXT,
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                appointment_type TEXT DEFAULT 'Product Demo',
                status TEXT DEFAULT 'Confirmed',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(appointment_date, appointment_time),
                FOREIGN KEY (session_id) REFERENCES leads(session_id)
            )
        """)

        # --- Call Summaries Table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS call_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                summary TEXT,
                customer_requirements TEXT,
                lead_classification TEXT,
                sentiment_overall TEXT,
                appointment_status TEXT,
                next_action TEXT,
                total_messages INTEGER DEFAULT 0,
                duration_seconds INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES leads(session_id)
            )
        """)

        logger.info("✅ Database initialized successfully")


# =====================================================
# LEAD OPERATIONS
# =====================================================

def save_lead(session_id: str, lead_data: Dict[str, Any]) -> bool:
    """
    Insert or update a lead record.
    Uses INSERT OR REPLACE to handle both new and existing leads.
    """
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO leads (
                    session_id, name, email, phone, company,
                    product_interest, budget, timeline, requirements,
                    lead_score, lead_classification, classification_reason,
                    sentiment, sentiment_score, appointment_booked,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    name = excluded.name,
                    email = excluded.email,
                    phone = excluded.phone,
                    company = excluded.company,
                    product_interest = excluded.product_interest,
                    budget = excluded.budget,
                    timeline = excluded.timeline,
                    requirements = excluded.requirements,
                    lead_score = excluded.lead_score,
                    lead_classification = excluded.lead_classification,
                    classification_reason = excluded.classification_reason,
                    sentiment = excluded.sentiment,
                    sentiment_score = excluded.sentiment_score,
                    appointment_booked = excluded.appointment_booked,
                    updated_at = excluded.updated_at
            """, (
                session_id,
                lead_data.get("name"),
                lead_data.get("email"),
                lead_data.get("phone"),
                lead_data.get("company"),
                lead_data.get("product_interest"),
                lead_data.get("budget"),
                lead_data.get("timeline"),
                lead_data.get("requirements"),
                lead_data.get("lead_score", 0),
                lead_data.get("lead_classification", "Cold Lead"),
                lead_data.get("classification_reason", ""),
                lead_data.get("sentiment", "Neutral"),
                lead_data.get("sentiment_score", 0.0),
                1 if lead_data.get("appointment_booked") else 0,
                datetime.now().isoformat()
            ))
        return True
    except Exception as e:
        logger.error(f"Error saving lead: {e}")
        return False


def get_all_leads() -> List[Dict]:
    """Retrieve all leads as a list of dictionaries."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM leads ORDER BY created_at DESC
            """).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching leads: {e}")
        return []


def get_lead_by_session(session_id: str) -> Optional[Dict]:
    """Get a single lead by session ID."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM leads WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error fetching lead: {e}")
        return None


def get_dashboard_stats() -> Dict[str, Any]:
    """
    Returns aggregated statistics for the dashboard page.
    Single query approach for performance.
    """
    try:
        with get_connection() as conn:
            # Lead counts by classification
            leads_data = conn.execute("""
                SELECT
                    COUNT(*) as total_leads,
                    SUM(CASE WHEN lead_classification = 'Hot Lead' THEN 1 ELSE 0 END) as hot_leads,
                    SUM(CASE WHEN lead_classification = 'Warm Lead' THEN 1 ELSE 0 END) as warm_leads,
                    SUM(CASE WHEN lead_classification = 'Cold Lead' THEN 1 ELSE 0 END) as cold_leads,
                    SUM(CASE WHEN appointment_booked = 1 THEN 1 ELSE 0 END) as appointments_booked
                FROM leads
            """).fetchone()

            # Total conversations
            conv_count = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM conversations"
            ).fetchone()[0]

            # Sentiment distribution
            sentiment_data = conn.execute("""
                SELECT sentiment, COUNT(*) as count
                FROM leads
                WHERE sentiment IS NOT NULL
                GROUP BY sentiment
            """).fetchall()

            # Recent leads (last 5)
            recent_leads = conn.execute("""
                SELECT name, lead_classification, sentiment, created_at
                FROM leads
                ORDER BY created_at DESC
                LIMIT 5
            """).fetchall()

            return {
                "total_leads": leads_data["total_leads"] or 0,
                "hot_leads": leads_data["hot_leads"] or 0,
                "warm_leads": leads_data["warm_leads"] or 0,
                "cold_leads": leads_data["cold_leads"] or 0,
                "appointments_booked": leads_data["appointments_booked"] or 0,
                "total_conversations": conv_count or 0,
                "sentiment_distribution": {row["sentiment"]: row["count"] for row in sentiment_data},
                "recent_leads": [dict(row) for row in recent_leads]
            }
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        return {
            "total_leads": 0, "hot_leads": 0, "warm_leads": 0,
            "cold_leads": 0, "appointments_booked": 0,
            "total_conversations": 0, "sentiment_distribution": {},
            "recent_leads": []
        }


# =====================================================
# CONVERSATION OPERATIONS
# =====================================================

def save_message(session_id: str, role: str, content: str, sentiment: str = "Neutral"):
    """Save a single conversation message to the database."""
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO conversations (session_id, role, content, sentiment)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, sentiment))
        return True
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return False


def get_conversation_history(session_id: str) -> List[Dict]:
    """Get the full conversation history for a session."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT role, content, sentiment, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,)).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        return []


def get_all_conversations() -> List[Dict]:
    """Get summary of all conversations with customer names."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    c.session_id,
                    l.name as customer_name,
                    l.lead_classification,
                    l.sentiment,
                    COUNT(c.id) as message_count,
                    MAX(c.timestamp) as last_message
                FROM conversations c
                LEFT JOIN leads l ON c.session_id = l.session_id
                GROUP BY c.session_id
                ORDER BY last_message DESC
            """).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return []


# =====================================================
# APPOINTMENT OPERATIONS
# =====================================================

def book_appointment(appointment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Books an appointment slot.
    Returns success status and message.
    Prevents double-booking via UNIQUE constraint on date+time.
    """
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO appointments (
                    session_id, customer_name, customer_email,
                    customer_phone, appointment_date, appointment_time,
                    appointment_type, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                appointment_data.get("session_id"),
                appointment_data.get("customer_name"),
                appointment_data.get("customer_email"),
                appointment_data.get("customer_phone"),
                appointment_data.get("date"),
                appointment_data.get("time"),
                appointment_data.get("type", "Product Demo"),
                appointment_data.get("notes", "")
            ))

            # Mark lead as having appointment booked
            if appointment_data.get("session_id"):
                conn.execute("""
                    UPDATE leads SET appointment_booked = 1, updated_at = ?
                    WHERE session_id = ?
                """, (datetime.now().isoformat(), appointment_data.get("session_id")))

        return {"success": True, "message": "Appointment booked successfully!"}

    except sqlite3.IntegrityError:
        return {"success": False, "message": "This slot is already booked. Please choose another time."}
    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return {"success": False, "message": f"Booking failed: {str(e)}"}


def get_booked_slots() -> List[str]:
    """Returns list of already booked date+time slots to prevent double booking."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT appointment_date || ' ' || appointment_time as slot
                FROM appointments
                WHERE status != 'Cancelled'
            """).fetchall()
            return [row["slot"] for row in rows]
    except Exception as e:
        logger.error(f"Error fetching booked slots: {e}")
        return []


def get_all_appointments() -> List[Dict]:
    """Retrieve all appointments for the appointments management page."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM appointments ORDER BY appointment_date ASC, appointment_time ASC
            """).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching appointments: {e}")
        return []


# =====================================================
# CALL SUMMARY OPERATIONS
# =====================================================

def save_call_summary(session_id: str, summary_data: Dict[str, Any]) -> bool:
    """Save the AI-generated call summary after conversation ends."""
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO call_summaries (
                    session_id, summary, customer_requirements,
                    lead_classification, sentiment_overall,
                    appointment_status, next_action,
                    total_messages, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    customer_requirements = excluded.customer_requirements,
                    lead_classification = excluded.lead_classification,
                    sentiment_overall = excluded.sentiment_overall,
                    appointment_status = excluded.appointment_status,
                    next_action = excluded.next_action,
                    total_messages = excluded.total_messages,
                    duration_seconds = excluded.duration_seconds
            """, (
                session_id,
                summary_data.get("summary"),
                summary_data.get("customer_requirements"),
                summary_data.get("lead_classification"),
                summary_data.get("sentiment_overall"),
                summary_data.get("appointment_status"),
                summary_data.get("next_action"),
                summary_data.get("total_messages", 0),
                summary_data.get("duration_seconds", 0)
            ))
        return True
    except Exception as e:
        logger.error(f"Error saving call summary: {e}")
        return False


def get_call_summaries() -> List[Dict]:
    """Get all call summaries with customer info joined."""
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    cs.*,
                    l.name as customer_name,
                    l.email as customer_email,
                    l.product_interest
                FROM call_summaries cs
                LEFT JOIN leads l ON cs.session_id = l.session_id
                ORDER BY cs.created_at DESC
            """).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching call summaries: {e}")
        return []
