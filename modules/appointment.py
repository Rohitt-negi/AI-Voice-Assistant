"""
=====================================================
Module: appointment.py
Purpose: Handle appointment slot generation,
         availability checking and booking logic.
         All appointments are stored in SQLite.
         No external calendar API required.
=====================================================
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# --- Configuration ---
# Appointment types available
APPOINTMENT_TYPES = [
    "Product Demo",
    "Sales Consultation",
    "Technical Walkthrough",
    "Pricing Discussion",
    "Free Trial Kickoff"
]

# Business hours for appointment generation
BUSINESS_START_HOUR = 9   # 9:00 AM
BUSINESS_END_HOUR = 18    # 6:00 PM
SLOT_DURATION_MINUTES = 30  # Each slot is 30 minutes


def generate_available_slots(days_ahead: int = 7) -> List[Dict[str, str]]:
    """
    Generate appointment time slots for the next N business days.
    Excludes weekends (Saturday=5, Sunday=6).

    Returns list of dicts: {date, time, label, datetime_key}
    """
    slots = []
    current_date = date.today()

    # Import here to avoid circular import
    from modules.database import get_booked_slots
    booked = set(get_booked_slots())

    days_added = 0
    offset = 1  # Start from tomorrow

    while days_added < days_ahead:
        check_date = current_date + timedelta(days=offset)
        offset += 1

        # Skip weekends
        if check_date.weekday() >= 5:
            continue

        days_added += 1
        date_str = check_date.strftime("%Y-%m-%d")
        date_label = check_date.strftime("%A, %d %B %Y")

        # Generate time slots for this day
        current_hour = BUSINESS_START_HOUR
        current_min = 0

        while (current_hour * 60 + current_min) < (BUSINESS_END_HOUR * 60):
            time_str = f"{current_hour:02d}:{current_min:02d}"
            datetime_key = f"{date_str} {time_str}"

            # 12-hour format for display
            period = "AM" if current_hour < 12 else "PM"
            display_hour = current_hour if current_hour <= 12 else current_hour - 12
            if display_hour == 0:
                display_hour = 12
            time_label = f"{display_hour}:{current_min:02d} {period}"

            is_available = datetime_key not in booked

            slots.append({
                "date": date_str,
                "time": time_str,
                "time_label": time_label,
                "date_label": date_label,
                "label": f"{date_label} at {time_label}",
                "datetime_key": datetime_key,
                "available": is_available
            })

            # Advance by slot duration
            current_min += SLOT_DURATION_MINUTES
            if current_min >= 60:
                current_hour += current_min // 60
                current_min = current_min % 60

    return slots


def get_available_slots_only(days_ahead: int = 7) -> List[Dict[str, str]]:
    """Returns only the available (not booked) slots."""
    return [s for s in generate_available_slots(days_ahead) if s["available"]]


def format_slots_for_display(slots: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group slots by date for a cleaner display in the UI.

    Returns: { "Monday, 25 November 2025": [slot1, slot2, ...], ... }
    """
    grouped = {}
    for slot in slots:
        date_label = slot["date_label"]
        if date_label not in grouped:
            grouped[date_label] = []
        grouped[date_label].append(slot)
    return grouped


def validate_appointment_data(appointment_data: Dict) -> Dict[str, str]:
    """
    Validate appointment booking data before saving.
    Returns dict of field -> error message (empty if valid).
    """
    errors = {}

    if not appointment_data.get("customer_name", "").strip():
        errors["customer_name"] = "Customer name is required"

    if not appointment_data.get("customer_email", "").strip():
        errors["customer_email"] = "Email address is required"
    elif "@" not in appointment_data.get("customer_email", ""):
        errors["customer_email"] = "Please enter a valid email address"

    if not appointment_data.get("date"):
        errors["date"] = "Please select a date"

    if not appointment_data.get("time"):
        errors["time"] = "Please select a time slot"

    if not appointment_data.get("type"):
        errors["type"] = "Please select appointment type"

    return errors


def get_appointment_confirmation_message(appointment_data: Dict) -> str:
    """
    Generate a formatted confirmation message for a booked appointment.
    """
    name = appointment_data.get("customer_name", "Customer")
    appt_type = appointment_data.get("type", "Product Demo")
    date_str = appointment_data.get("date", "")
    time_str = appointment_data.get("time", "")

    # Format date nicely
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        date_display = date_obj.strftime("%A, %d %B %Y")
    except Exception:
        date_display = date_str

    # Format time nicely
    try:
        time_obj = datetime.strptime(time_str, "%H:%M")
        time_display = time_obj.strftime("%I:%M %p")
    except Exception:
        time_display = time_str

    return (
        f"🎉 Appointment Confirmed!\n\n"
        f"Dear {name},\n\n"
        f"Your **{appt_type}** has been successfully scheduled.\n\n"
        f"📅 **Date:** {date_display}\n"
        f"⏰ **Time:** {time_display} IST\n"
        f"📍 **Format:** Google Meet (link will be shared via email)\n\n"
        f"Our sales representative will reach out to confirm and share the meeting link.\n"
        f"Feel free to contact us at **sales@fortivex.ai** if you need to reschedule.\n\n"
        f"We look forward to speaking with you!"
    )
