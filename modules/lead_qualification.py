"""
=====================================================
Module: lead_qualification.py
Purpose: Automatically score and classify leads
         based on information collected during
         the sales conversation.

Lead Classification:
  Hot Lead  → Score 70-100 (Ready to buy, high intent)
  Warm Lead → Score 40-69  (Interested, needs nurturing)
  Cold Lead → Score 0-39   (Early stage, low intent)
=====================================================
"""

import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


# --- Scoring Weights ---
# Each collected data point adds to the lead score
SCORING_RULES = {
    "has_name": 10,
    "has_email": 15,
    "has_phone": 15,
    "has_company": 10,
    "has_product_interest": 15,
    "has_budget": 10,
    "has_timeline": 10,
    "has_requirements": 5,
    # Timeline bonuses (how urgently they want to buy)
    "timeline_immediate": 15,     # "this week", "immediately", "ASAP"
    "timeline_short": 10,         # "this month", "next month"
    "timeline_medium": 5,         # "in 3 months", "this quarter"
    # Budget bonuses
    "budget_above_50k": 10,
    "budget_10k_50k": 5,
    # Engagement bonuses
    "wants_demo": 10,
    "multiple_products": 5,
}


def _extract_budget_value(budget_str: str) -> int:
    """
    Try to extract a numeric value from a budget string.
    Handles formats like "50000", "50k", "₹50,000", "5 lakhs"
    Returns the value in rupees (integer).
    """
    if not budget_str:
        return 0

    budget_lower = budget_str.lower().strip()

    # Remove currency symbols and commas
    budget_clean = re.sub(r'[₹$,\s]', '', budget_lower)

    # Handle "lakhs" / "lakh"
    lakh_match = re.search(r'(\d+\.?\d*)\s*lakh', budget_lower)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)

    # Handle "k" (thousands)
    k_match = re.search(r'(\d+\.?\d*)\s*k', budget_clean)
    if k_match:
        return int(float(k_match.group(1)) * 1000)

    # Plain number
    num_match = re.search(r'(\d+)', budget_clean)
    if num_match:
        return int(num_match.group(1))

    return 0


def _get_timeline_category(timeline_str: str) -> str:
    """
    Categorize the purchase timeline from a text description.
    Returns: 'immediate', 'short', 'medium', 'long', or 'unknown'
    """
    if not timeline_str:
        return "unknown"

    tl = timeline_str.lower()

    immediate_keywords = ["immediately", "asap", "urgent", "right away", "this week", "today", "now"]
    short_keywords = ["this month", "next month", "within a month", "30 days", "soon"]
    medium_keywords = ["3 months", "quarter", "q1", "q2", "q3", "q4", "3-6 months"]
    long_keywords = ["6 months", "next year", "next quarter", "long term", "eventually", "someday"]

    for kw in immediate_keywords:
        if kw in tl:
            return "immediate"
    for kw in short_keywords:
        if kw in tl:
            return "short"
    for kw in medium_keywords:
        if kw in tl:
            return "medium"
    for kw in long_keywords:
        if kw in tl:
            return "long"

    return "unknown"


def calculate_lead_score(customer_info: Dict[str, Any], conversation_text: str = "") -> Dict[str, Any]:
    """
    Calculate the lead score and classification based on collected information.

    Args:
        customer_info: Dictionary with customer data (name, email, phone, etc.)
        conversation_text: Full conversation text for additional signal extraction

    Returns:
        Dict with:
        - score: integer 0-100
        - classification: 'Hot Lead', 'Warm Lead', or 'Cold Lead'
        - reason: human-readable explanation
        - breakdown: score breakdown per criterion
    """
    score = 0
    breakdown = []
    reasons = []

    # --- Basic Information Scoring ---
    if customer_info.get("name", "").strip():
        score += SCORING_RULES["has_name"]
        breakdown.append(f"+{SCORING_RULES['has_name']} Name provided")

    if customer_info.get("email", "").strip():
        score += SCORING_RULES["has_email"]
        breakdown.append(f"+{SCORING_RULES['has_email']} Email provided")
        reasons.append("contact email collected")

    if customer_info.get("phone", "").strip():
        score += SCORING_RULES["has_phone"]
        breakdown.append(f"+{SCORING_RULES['has_phone']} Phone provided")
        reasons.append("phone number collected")

    if customer_info.get("company", "").strip():
        score += SCORING_RULES["has_company"]
        breakdown.append(f"+{SCORING_RULES['has_company']} Company provided")
        reasons.append("business lead")

    if customer_info.get("product_interest", "").strip():
        score += SCORING_RULES["has_product_interest"]
        breakdown.append(f"+{SCORING_RULES['has_product_interest']} Product interest identified")
        reasons.append(f"interested in {customer_info.get('product_interest')}")

    if customer_info.get("requirements", "").strip():
        score += SCORING_RULES["has_requirements"]
        breakdown.append(f"+{SCORING_RULES['has_requirements']} Requirements shared")

    # --- Budget Scoring ---
    budget_str = customer_info.get("budget", "")
    if budget_str:
        score += SCORING_RULES["has_budget"]
        breakdown.append(f"+{SCORING_RULES['has_budget']} Budget mentioned")
        reasons.append("budget disclosed")

        budget_value = _extract_budget_value(budget_str)
        if budget_value >= 50000:
            score += SCORING_RULES["budget_above_50k"]
            breakdown.append(f"+{SCORING_RULES['budget_above_50k']} High budget (≥₹50K)")
            reasons.append("high budget")
        elif budget_value >= 10000:
            score += SCORING_RULES["budget_10k_50k"]
            breakdown.append(f"+{SCORING_RULES['budget_10k_50k']} Medium budget (₹10K-₹50K)")

    # --- Timeline Scoring ---
    timeline_str = customer_info.get("timeline", "")
    if timeline_str:
        score += SCORING_RULES["has_timeline"]
        breakdown.append(f"+{SCORING_RULES['has_timeline']} Timeline specified")

        timeline_cat = _get_timeline_category(timeline_str)
        if timeline_cat == "immediate":
            score += SCORING_RULES["timeline_immediate"]
            breakdown.append(f"+{SCORING_RULES['timeline_immediate']} Immediate purchase intent")
            reasons.append("wants to purchase immediately")
        elif timeline_cat == "short":
            score += SCORING_RULES["timeline_short"]
            breakdown.append(f"+{SCORING_RULES['timeline_short']} Short-term purchase timeline")
            reasons.append("planning to buy within a month")
        elif timeline_cat == "medium":
            score += SCORING_RULES["timeline_medium"]
            breakdown.append(f"+{SCORING_RULES['timeline_medium']} Medium-term timeline (3 months)")

    # --- Conversation Analysis ---
    if conversation_text:
        conv_lower = conversation_text.lower()

        # Check for demo interest
        demo_keywords = ["demo", "trial", "free trial", "show me", "demonstrate", "book", "schedule"]
        if any(kw in conv_lower for kw in demo_keywords):
            score += SCORING_RULES["wants_demo"]
            breakdown.append(f"+{SCORING_RULES['wants_demo']} Requested demo/trial")
            reasons.append("requested a product demo")

        # Check if multiple products mentioned
        products = ["voiceagent", "insightiq", "docusense", "chatassist"]
        mentioned = sum(1 for p in products if p in conv_lower)
        if mentioned > 1:
            score += SCORING_RULES["multiple_products"]
            breakdown.append(f"+{SCORING_RULES['multiple_products']} Interested in multiple products")

    # Cap score at 100
    score = min(score, 100)

    # --- Classify Lead ---
    if score >= 70:
        classification = "Hot Lead"
        emoji = "🔥"
    elif score >= 40:
        classification = "Warm Lead"
        emoji = "🌡️"
    else:
        classification = "Cold Lead"
        emoji = "❄️"

    # Build explanation
    if reasons:
        reason_text = f"{emoji} {classification}: Scored {score}/100 because customer {', '.join(reasons[:3])}."
    else:
        reason_text = f"{emoji} {classification}: Scored {score}/100. Insufficient information collected yet."

    return {
        "score": score,
        "classification": classification,
        "reason": reason_text,
        "breakdown": breakdown
    }


def get_lead_color(classification: str) -> str:
    """Returns a UI color for the lead classification badge."""
    return {
        "Hot Lead": "#E74C3C",
        "Warm Lead": "#F39C12",
        "Cold Lead": "#3498DB"
    }.get(classification, "#95A5A6")


def get_next_action(classification: str, appointment_booked: bool) -> str:
    """
    Suggests the next action for the sales team based on lead classification.
    """
    if appointment_booked:
        return "✅ Demo/consultation scheduled — prepare product presentation and confirm appointment 24hrs prior."

    actions = {
        "Hot Lead": "🔥 High priority! Call within 2 hours. Send personalized proposal and demo link immediately.",
        "Warm Lead": "🌡️ Follow up within 24 hours. Send product brochure and schedule a demo call.",
        "Cold Lead": "❄️ Add to nurture sequence. Send introductory email with case studies and check back in 2 weeks."
    }
    return actions.get(classification, "Contact customer for more information.")
