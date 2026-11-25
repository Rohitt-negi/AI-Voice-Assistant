
"""
=====================================================
Module: text_to_speech.py
Purpose: Multilingual AI text-to-speech using Edge TTS.

Features:
    - Automatic language detection from response text
    - English, Hindi, Punjabi and regional languages
    - Automatic voice switching
    - Male and female voice selection
    - Streamlit compatibility
    - Offline pyttsx3 fallback
    - No additional paid TTS API key required
=====================================================
"""

import asyncio
import os
import re
import logging
import tempfile
import concurrent.futures
from typing import Optional

logger = logging.getLogger(__name__)


# =====================================================
# TTS CONFIGURATION
# =====================================================

TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")


# =====================================================
# AVAILABLE VOICES FOR STREAMLIT UI
# =====================================================

AVAILABLE_VOICES = {
    "Aria (US Female) - Recommended": "en-US-AriaNeural",
    "Guy (US Male)": "en-US-GuyNeural",
    "Jenny (US Female)": "en-US-JennyNeural",
    "Sonia (UK Female)": "en-GB-SoniaNeural",
    "Ryan (UK Male)": "en-GB-RyanNeural",

    # Indian English
    "Neerja (Indian Female)": "en-IN-NeerjaNeural",
    "Prabhat (Indian Male)": "en-IN-PrabhatNeural",

    # Hindi
    "Swara (Hindi Female)": "hi-IN-SwaraNeural",
    "Madhur (Hindi Male)": "hi-IN-MadhurNeural",

    # Punjabi
    "Vaani (Punjabi Female)": "pa-IN-VaaniNeural",
    "Ojas (Punjabi Male)": "pa-IN-OjasNeural",

    # Tamil
    "Pallavi (Tamil Female)": "ta-IN-PallaviNeural",
    "Valluvar (Tamil Male)": "ta-IN-ValluvarNeural",

    # Telugu
    "Shruti (Telugu Female)": "te-IN-ShrutiNeural",
    "Mohan (Telugu Male)": "te-IN-MohanNeural",

    # Marathi
    "Aarohi (Marathi Female)": "mr-IN-AarohiNeural",
    "Manohar (Marathi Male)": "mr-IN-ManoharNeural",

    # Gujarati
    "Dhwani (Gujarati Female)": "gu-IN-DhwaniNeural",
    "Niranjan (Gujarati Male)": "gu-IN-NiranjanNeural",

    # Bengali
    "Tanishaa (Bengali Female)": "bn-IN-TanishaaNeural",
    "Bashkar (Bengali Male)": "bn-IN-BashkarNeural",

    # Kannada
    "Sapna (Kannada Female)": "kn-IN-SapnaNeural",
    "Gagan (Kannada Male)": "kn-IN-GaganNeural",

    # Malayalam
    "Sobhana (Malayalam Female)": "ml-IN-SobhanaNeural",
    "Midhun (Malayalam Male)": "ml-IN-MidhunNeural",

    # Urdu
    "Uzma (Urdu Female)": "ur-PK-UzmaNeural",
    "Asad (Urdu Male)": "ur-PK-AsadNeural",
}


# =====================================================
# LANGUAGE TO VOICE MAPPING
# =====================================================

LANGUAGE_VOICES = {
    "en": ("en-IN-NeerjaNeural", "en-IN-PrabhatNeural"),
    "hi": ("hi-IN-SwaraNeural", "hi-IN-MadhurNeural"),
    "pa": ("pa-IN-VaaniNeural", "pa-IN-OjasNeural"),
    "ta": ("ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"),
    "te": ("te-IN-ShrutiNeural", "te-IN-MohanNeural"),
    "mr": ("mr-IN-AarohiNeural", "mr-IN-ManoharNeural"),
    "gu": ("gu-IN-DhwaniNeural", "gu-IN-NiranjanNeural"),
    "bn": ("bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"),
    "kn": ("kn-IN-SapnaNeural", "kn-IN-GaganNeural"),
    "ml": ("ml-IN-SobhanaNeural", "ml-IN-MidhunNeural"),
    "ur": ("ur-PK-UzmaNeural", "ur-PK-AsadNeural"),
}


# =====================================================
# AUTOMATIC LANGUAGE DETECTION
# =====================================================

def detect_language(text: str) -> str:
    """
    Detect the language from the AI-generated response
    using Unicode script ranges.

    Devanagari defaults to Hindi unless Marathi is
    explicitly selected.
    """

    if not text:
        return "en"

    patterns = {
        "hi": r"[\u0900-\u097F]",
        "pa": r"[\u0A00-\u0A7F]",
        "gu": r"[\u0A80-\u0AFF]",
        "bn": r"[\u0980-\u09FF]",
        "ta": r"[\u0B80-\u0BFF]",
        "te": r"[\u0C00-\u0C7F]",
        "kn": r"[\u0C80-\u0CFF]",
        "ml": r"[\u0D00-\u0D7F]",
        "ur": r"[\u0600-\u06FF]",
    }

    scores = {
        language: len(re.findall(pattern, text))
        for language, pattern in patterns.items()
    }

    detected = max(scores, key=scores.get)

    if scores[detected] > 0:
        return detected

    return "en"


# =====================================================
# AUTOMATIC VOICE SELECTION
# =====================================================

def select_voice(
    text: str,
    preferred_voice: Optional[str] = None,
    language: Optional[str] = None
) -> str:
    """
    Automatically select a compatible voice based on
    the language of the AI response.

    Preserves the user's male/female voice preference.
    """

    preferred_voice = preferred_voice or TTS_VOICE

    # Accept either a UI label or a voice ID
    preferred_voice = AVAILABLE_VOICES.get(
        preferred_voice,
        preferred_voice
    )

    preferred_language = preferred_voice.split("-")[0]

    detected_language = detect_language(text)

    # Optional explicit language override
    if language:
        detected_language = language.lower().split("-")[0]

    # Marathi and Hindi share the Devanagari script
    elif detected_language == "hi" and preferred_language == "mr":
        detected_language = "mr"

    # Preserve manually selected voice if compatible
    if preferred_language == detected_language:
        return preferred_voice

    male_voices = {
        "en-US-GuyNeural",
        "en-GB-RyanNeural",
    }

    male_voices.update(
        voices[1] for voices in LANGUAGE_VOICES.values()
    )

    is_male = preferred_voice in male_voices

    voices = LANGUAGE_VOICES.get(
        detected_language,
        LANGUAGE_VOICES["en"]
    )

    selected_voice = voices[1] if is_male else voices[0]

    logger.info(
        f"🌐 Detected language: {detected_language} | "
        f"Selected voice: {selected_voice}"
    )

    return selected_voice


# =====================================================
# CLEAN AI RESPONSE FOR SPEECH
# =====================================================

def clean_text_for_tts(text: str) -> str:
    """
    Remove markdown and formatting without removing
    Hindi, Punjabi or other regional language characters.
    """

    if not text:
        return ""

    # Remove fenced code blocks
    text = re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove markdown bold and italic
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)

    # Remove markdown headers
    text = re.sub(
        r"^#{1,6}\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove markdown links, retaining link text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove bullet points
    text = re.sub(
        r"^\s*[-*•]\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove numbered list markers
    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove inline code markers
    text = text.replace("`", "")

    # Handle currency according to language
    language = detect_language(text)

    currency_words = {
        "hi": " रुपये ",
        "pa": " ਰੁਪਏ ",
        "mr": " रुपये ",
        "en": " rupees ",
    }

    text = text.replace(
        "₹",
        currency_words.get(language, " rupees ")
    )

    text = text.replace("@", " at ")

    # Normalize whitespace
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# EDGE TTS SYNTHESIS
# =====================================================

async def _synthesize_edge_tts(
    text: str,
    output_path: str,
    voice: str,
    rate: str,
    volume: str
) -> bool:

    try:
        import edge_tts

        logger.info(f"🔊 Generating speech using: {voice}")

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume
        )

        await communicate.save(output_path)

        return (
            os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        )

    except Exception as e:
        logger.error(f"❌ Edge TTS synthesis failed: {e}")
        return False


# =====================================================
# ASYNC EXECUTION FOR STREAMLIT
# =====================================================

def _run_async(coroutine):
    """
    Safely execute an async function from Streamlit.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    ) as executor:

        future = executor.submit(
            asyncio.run,
            coroutine
        )

        return future.result()


# =====================================================
# OFFLINE TTS FALLBACK
# =====================================================

def _offline_tts(
    text: str,
    output_path: str,
    language: str
) -> Optional[str]:

    try:
        import pyttsx3

        engine = pyttsx3.init()

        voices = engine.getProperty("voices")

        language_names = {
            "en": "english",
            "hi": "hindi",
            "pa": "punjabi",
            "ta": "tamil",
            "te": "telugu",
            "mr": "marathi",
            "gu": "gujarati",
            "bn": "bengali",
            "kn": "kannada",
            "ml": "malayalam",
            "ur": "urdu",
        }

        target = language_names.get(language, "english")

        matching_voice = None

        for v in voices:

            metadata = (
                str(v.name) + " " +
                str(v.id) + " " +
                str(getattr(v, "languages", []))
            ).lower()

            if target in metadata:
                matching_voice = v.id
                break

        if matching_voice:
            engine.setProperty("voice", matching_voice)

        elif language != "en":
            logger.warning(
                f"No offline voice installed for {language}"
            )
            return None

        engine.setProperty("rate", 160)
        engine.setProperty("volume", 0.9)

        # macOS commonly uses AIFF for native speech output
        import sys

        extension = ".aiff" if sys.platform == "darwin" else ".wav"

        fallback_path = (
            os.path.splitext(output_path)[0] + extension
        )

        engine.save_to_file(text, fallback_path)
        engine.runAndWait()

        if (
            os.path.exists(fallback_path)
            and os.path.getsize(fallback_path) > 0
        ):
            logger.info(
                f"✅ Offline TTS saved: {fallback_path}"
            )
            return fallback_path

    except Exception as e:
        logger.error(f"❌ Offline TTS failed: {e}")

    return None


# =====================================================
# MAIN TEXT-TO-SPEECH FUNCTION
# =====================================================

def synthesize_speech(
    text: str,
    output_path: Optional[str] = None,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    volume: Optional[str] = None,
    language: Optional[str] = None
) -> Optional[str]:
    """
    Convert AI-generated text into multilingual speech.

    Automatically switches voices according to the
    response language.

    Returns:
        Path to generated audio file, or None on failure.
    """

    if not text or not text.strip():
        return None

    # Clean response without losing regional characters
    text = clean_text_for_tts(text)

    if not text:
        return None

    # IMPORTANT: Automatically select the correct voice
    selected_voice = select_voice(
        text=text,
        preferred_voice=voice,
        language=language
    )

    detected_language = selected_voice.split("-")[0]

    rate = rate or TTS_RATE
    volume = volume or TTS_VOLUME

    logger.info(f"🌐 Response language: {detected_language}")
    logger.info(f"🎙️ Active TTS voice: {selected_voice}")

    # Create temporary MP3 file
    temporary_output = output_path is None

    if temporary_output:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as tmp:
            output_path = tmp.name

    # Primary method: Microsoft Edge TTS
    try:
        success = _run_async(
            _synthesize_edge_tts(
                text=text,
                output_path=output_path,
                voice=selected_voice,
                rate=rate,
                volume=volume
            )
        )

        if success:
            logger.info(f"✅ TTS audio saved: {output_path}")
            return output_path

    except Exception as e:
        logger.warning(
            f"Edge TTS failed: {e}. Trying offline fallback."
        )

    # Secondary method: Offline TTS
    fallback_path = _offline_tts(
        text=text,
        output_path=output_path,
        language=detected_language
    )

    # Remove unsuccessful temporary MP3
    if temporary_output:
        try:
            os.unlink(output_path)
        except OSError:
            pass

    return fallback_path
