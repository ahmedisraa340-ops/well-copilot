"""Gemini wrapper (Google AI Studio key). Plain REST, so no extra SDK is needed.
ask() returns None on any failure, and every caller has a rule-based fallback."""
import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_last_error = ""


def _key():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        try:  # Streamlit Cloud secrets
            import streamlit as st
            key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return (key or "").strip().strip('"').strip("'") or None


def available():
    return _key() is not None


def last_error():
    return _last_error


def ask(system, user, max_tokens=1200):
    """Return model text, or None if there is no key or the call fails."""
    global _last_error
    key = _key()
    if not key:
        return None
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        # extra headroom: Gemini 2.5 models spend part of this budget on internal reasoning
        "generationConfig": {"maxOutputTokens": max_tokens * 4, "temperature": 0.3},
    }
    try:
        r = requests.post(ENDPOINT.format(model=MODEL), json=body, timeout=90,
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError("empty response")
        _last_error = ""
        return text
    except Exception as e:
        _last_error = str(e)
        print(f"[llm] using rule-based text instead: {e}")
        return None
