
"""
=====================================================
Module: speech_to_text.py
Purpose: Multilingual speech recognition using local
         OpenAI Whisper (free).

Features:
    - Automatic language detection
    - English, Hindi, Punjabi, Hinglish and more
    - Microphone recording
    - Audio file transcription
    - Language tracking for multilingual TTS
    - Error handling and temporary file cleanup
=====================================================
"""

import os
import logging
import tempfile
import threading
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# --- Whisper configuration ---
_whisper_model = None
_model_lock = threading.Lock()
_transcription_state = threading.local()

# Multilingual Whisper model
# Options: tiny, base, small, medium, large
WHISPER_MODEL_SIZE = "base"

# None = automatic language detection
# "en" = English
# "hi" = Hindi
# "pa" = Punjabi
# "ur" = Urdu
DEFAULT_LANGUAGE = None


def _load_whisper():
    """
    Lazy-load the multilingual Whisper model.
    """

    global _whisper_model

    with _model_lock:
        if _whisper_model is not None:
            return True

        try:
            import whisper

            logger.info(
                f"Loading Whisper '{WHISPER_MODEL_SIZE}' model..."
            )

            _whisper_model = whisper.load_model(
                WHISPER_MODEL_SIZE
            )

            logger.info("Whisper model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return False


def get_detected_language() -> Optional[str]:
    """
    Return the language detected during the latest
    transcription in the current thread.

    Examples:
        en = English
        hi = Hindi
        pa = Punjabi
        ur = Urdu
    """

    return getattr(
        _transcription_state, "language", None
    )


def transcribe_audio_file(
    audio_path: str,
    language: Optional[str] = DEFAULT_LANGUAGE
) -> Optional[str]:
    """
    Transcribe audio using multilingual Whisper.

    language=None enables automatic detection.

    Returns:
        Transcribed text or None on failure.
    """

    _transcription_state.language = None

    if not _load_whisper():
        return None

    try:
        logger.info(f"Transcribing: {audio_path}")

        result = _whisper_model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            fp16=False,
            verbose=False,
            temperature=0.0
        )

        text = result.get("text", "").strip()

        _transcription_state.language = result.get(
            "language", language
        )

        logger.info(
            f"Detected language: {get_detected_language()}"
        )

        logger.info(f"Transcription result: '{text}'")

        return text if text else None

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


def transcribe_audio_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    language: Optional[str] = DEFAULT_LANGUAGE
) -> Optional[str]:
    """
    Convert raw float32 microphone audio bytes
    into multilingual text.
    """

    _transcription_state.language = None
    tmp_path = None

    try:
        import soundfile as sf

        if not audio_bytes:
            return None

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        audio_array = np.frombuffer(
            audio_bytes,
            dtype=np.float32
        )

        sf.write(
            tmp_path,
            audio_array,
            sample_rate
        )

        return transcribe_audio_file(
            tmp_path,
            language=language
        )

    except Exception as e:
        logger.error(
            f"Audio bytes transcription failed: {e}"
        )
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def record_audio(
    duration_seconds: int = 5,
    sample_rate: int = 16000
) -> Optional[bytes]:
    """
    Record audio from the system microphone.
    """

    try:
        import sounddevice as sd

        logger.info(
            f"Recording for {duration_seconds} seconds..."
        )

        audio_data = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )

        sd.wait()

        logger.info("Recording complete")

        return audio_data.tobytes()

    except Exception as e:
        logger.error(
            f"Microphone recording failed: {e}"
        )
        return None


def record_and_transcribe(
    duration_seconds: int = 5,
    language: Optional[str] = DEFAULT_LANGUAGE
) -> Optional[str]:
    """
    Record microphone audio and automatically
    detect and transcribe the spoken language.
    """

    _transcription_state.language = None

    audio_bytes = record_audio(
        duration_seconds=duration_seconds
    )

    if audio_bytes is None:
        return None

    return transcribe_audio_bytes(
        audio_bytes,
        language=language
    )


def is_microphone_available() -> bool:
    """
    Check whether a microphone is accessible.
    """

    try:
        import sounddevice as sd

        devices = sd.query_devices()

        input_devices = [
            device for device in devices
            if device["max_input_channels"] > 0
        ]

        return len(input_devices) > 0

    except Exception:
        return False
