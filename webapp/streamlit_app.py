"""Conversational price estimator - Streamlit + Google Gemini (free tier).

Gemini chats with the user (in English or Spanish, matching how they write),
collects the flat's features including the Madrid district, and calls the
``estimate_price`` tool. That tool returns a "district €/m² x m2" estimate
(~2024 figures from ``data/precio_m2_distrito.csv``) plus the thesis network's
number from ``docs/model.json`` (NumPy, no TensorFlow) as a reference.

Deploy on Streamlit Community Cloud with the main file set to
``webapp/streamlit_app.py`` and a ``GEMINI_API_KEY`` secret (free key at
https://aistudio.google.com/apikey).

Run locally:  GEMINI_API_KEY=...  streamlit run webapp/streamlit_app.py
"""

from __future__ import annotations

import os
import re
import time

import google.generativeai as genai
import streamlit as st

from pricing import estimate_by_district, load_districts, load_model, predict_price

# gemini-*-lite models get the most generous free-tier request quota, which
# matters here: automatic function calling makes 2-3 API calls per user turn.
# Pin another with a GEMINI_MODEL secret / env var or the sidebar picker.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_RETRIES = 2

st.set_page_config(page_title="Madrid flat price chat", page_icon="🏠")
MODEL = load_model()
DISTRICTS = load_districts()

SYSTEM = (
    "You help estimate the sale price of a flat in Madrid. Reply in the same "
    "language the user writes in (English or Spanish); keep answers short. You "
    "need: the Madrid district, number of rooms, number of bathrooms, floor area "
    "in m2, whether it has parking (yes/no) and how many schools are nearby (if "
    f"the user doesn't know, use 9 and say so). Valid districts: {', '.join(DISTRICTS)}. "
    "Ask for whatever is missing, accept values in any order, and once you have "
    "them all call the estimate_price function. Report the euro figures: the main "
    "one (district €/m2, ~2024 prices) and mention the thesis neural-network "
    "reference (~2020 data). Make clear these are rough estimates, not an appraisal."
)

STR = {
    "English": {
        "title": "🏠 Madrid flat price chat",
        "caption": "€/m² by district (~2024) + thesis neural network (~2020) · rough estimate",
        "no_key": "Missing `GEMINI_API_KEY` (app secret or environment variable).",
        "model_label": "Gemini model",
        "model_help": "The free tier caps requests per minute; on a 429, wait a few "
        "seconds or pick a `-lite` model.",
        "language_label": "Language / Idioma",
        "load_fail": "Could not load `{model}`: {exc}",
        "input": "Tell me about the flat…",
        "thinking": "Thinking…",
        "retry": "Free-tier limit hit, retrying in {wait:.0f} s…",
        "quota": "Gemini's free quota for this minute is used up. Wait a bit and try "
        "again, or pick a `-lite` model in the sidebar.",
        "models_available": "\n\nAvailable models: ",
        "gemini_error": "Error talking to Gemini: {exc}",
        "failed": "Could not complete the request.",
    },
    "Español": {
        "title": "🏠 Tasador conversacional de pisos",
        "caption": "€/m² por distrito (~2024) + red neuronal del TFM (~2020) · estimación aproximada",
        "no_key": "Falta `GEMINI_API_KEY` (secret de la app o variable de entorno).",
        "model_label": "Modelo Gemini",
        "model_help": "El tier gratuito limita las peticiones por minuto; si ves un "
        "error 429, espera unos segundos o elige un modelo `-lite`.",
        "language_label": "Idioma / Language",
        "load_fail": "No se pudo cargar `{model}`: {exc}",
        "input": "Cuéntame sobre el piso…",
        "thinking": "Pensando…",
        "retry": "Límite gratuito alcanzado, reintento en {wait:.0f} s…",
        "quota": "Se ha agotado la cuota gratuita de Gemini para este minuto. Espera "
        "un poco y vuelve a intentarlo, o elige un modelo `-lite` en la barra lateral.",
        "models_available": "\n\nModelos disponibles: ",
        "gemini_error": "Error al hablar con Gemini: {exc}",
        "failed": "No se pudo completar la petición.",
    },
}


def estimate_price(
    district: str, rooms: int, bathrooms: int, area_m2: float, parking: int, schools: int
) -> dict:
    """Estimate the sale price of a flat in Madrid.

    Args:
        district: Madrid district (e.g. Salamanca, Chamberí, Carabanchel).
        rooms: number of rooms.
        bathrooms: number of bathrooms.
        area_m2: floor area in square metres.
        parking: 1 if it has a parking space, 0 otherwise.
        schools: number of nearby schools (use 9 if unknown).
    """
    by_district = estimate_by_district(DISTRICTS, district, area_m2, parking)
    nn_price = predict_price(
        MODEL,
        {
            "Habitaciones": rooms,
            "Aseos": bathrooms,
            "Superficie": area_m2,
            "Parking": parking,
            "Colegios": schools,
        },
    )
    return {
        "price_eur": by_district["price_eur"],
        "method": f"{by_district['distrito']} at {by_district['eur_m2']:,} €/m² (~2024)",
        "reference_neural_network_2020_eur": round(nn_price),
    }


def _from_secret_or_env(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml at all
        pass
    return os.environ.get(name)


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    """Model names that support generateContent (for the picker / error hints)."""
    try:
        names = [
            m.name.removeprefix("models/")
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        return [n for n in names if "flash" in n or "pro" in n] or names
    except Exception:
        return []


def build_chat(model_name: str):
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM, tools=[estimate_price])
    return model.start_chat(enable_automatic_function_calling=True)


def _retry_after(exc: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc)) or re.search(r"seconds:\s*(\d+)", str(exc))
    return min(float(match.group(1)) + 1, 30) if match else 10.0


def send_message(chat, prompt: str, t: dict) -> str:
    """Send a turn, backing off once or twice on a 429 (free-tier rate limit)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat.send_message(prompt).text
        except Exception as exc:
            text = str(exc)
            if "429" in text and attempt < MAX_RETRIES:
                wait = _retry_after(exc)
                with st.spinner(t["retry"].format(wait=wait)):
                    time.sleep(wait)
                continue
            if "404" in text or "not found" in text.lower():
                models = available_models()
                hint = (t["models_available"] + ", ".join(models)) if models else ""
                return f"Error: {exc}{hint}"
            if "429" in text:
                return t["quota"]
            return t["gemini_error"].format(exc=exc)
    return t["failed"]


with st.sidebar:
    language = st.radio("Language / Idioma", list(STR), horizontal=True)
t = STR[language]

st.title(t["title"])
st.caption(t["caption"])

api_key = _from_secret_or_env("GEMINI_API_KEY")
if not api_key:
    st.error(t["no_key"])
    st.stop()
genai.configure(api_key=api_key)

preferred = _from_secret_or_env("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
options = available_models() or [preferred]
if preferred not in options:
    options = [preferred, *options]
with st.sidebar:
    model_name = st.selectbox(t["model_label"], options, index=options.index(preferred))
    st.caption(t["model_help"])

# (Re)build the chat when the app starts or the model changes.
if st.session_state.get("model_name") != model_name:
    try:
        st.session_state.chat = build_chat(model_name)
        st.session_state.model_name = model_name
    except Exception as exc:
        st.error(t["load_fail"].format(model=model_name, exc=exc))
        st.stop()
chat = st.session_state.chat

for message in chat.history:
    role = "assistant" if message.role == "model" else "user"
    text = "".join(getattr(part, "text", "") for part in message.parts)
    if text.strip():
        with st.chat_message(role):
            st.write(text)

if prompt := st.chat_input(t["input"]):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner(t["thinking"]):
            answer = send_message(chat, prompt, t)
        st.write(answer)
