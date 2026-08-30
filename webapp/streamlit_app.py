"""Conversational price estimator - Streamlit + Google Gemini (free tier).

Gemini chats with the user, collects the flat's features and calls the
``estimar_precio`` tool, which runs the model from ``docs/model.json`` locally
(NumPy, no TensorFlow). Deploy on Streamlit Community Cloud with the main file
set to ``webapp/streamlit_app.py`` and a ``GEMINI_API_KEY`` secret (get one free
at https://aistudio.google.com/apikey).

Run locally:  GEMINI_API_KEY=...  streamlit run webapp/streamlit_app.py
"""

from __future__ import annotations

import os
import re
import time

import google.generativeai as genai
import streamlit as st

from pricing import load_model, predict_price

# gemini-*-lite models get the most generous free-tier request quota, which
# matters here: automatic function calling makes 2-3 API calls per user turn.
# Pin another with a GEMINI_MODEL secret / env var or the sidebar picker.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_RETRIES = 2

st.set_page_config(page_title="Tasador conversacional", page_icon="🏠")
MODEL = load_model()

SYSTEM = (
    "Ayudas a estimar el precio de venta de un piso en Madrid. Habla en español, "
    "respuestas breves. Necesitas: número de habitaciones, número de aseos, "
    "superficie en m², si tiene parking (sí/no) y cuántos colegios hay cerca "
    "(si el usuario no lo sabe, usa 9 y díselo). Pregunta lo que falte, acepta "
    "los datos en cualquier orden, y cuando los tengas todos llama a la función "
    "estimar_precio y da el resultado en euros, aclarando que es una estimación "
    "aproximada del modelo, no una tasación."
)


def estimar_precio(habitaciones: int, aseos: int, superficie: float, parking: int, colegios: int) -> dict:
    """Estima el precio de venta de un piso en Madrid.

    Args:
        habitaciones: número de habitaciones.
        aseos: número de baños.
        superficie: superficie en metros cuadrados.
        parking: 1 si tiene plaza de garaje, 0 si no.
        colegios: número de colegios cercanos (usa 9 si se desconoce).
    """
    price = predict_price(
        MODEL,
        {
            "Habitaciones": habitaciones,
            "Aseos": aseos,
            "Superficie": superficie,
            "Parking": parking,
            "Colegios": colegios,
        },
    )
    return {"precio_estimado_eur": round(price)}


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
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM, tools=[estimar_precio])
    return model.start_chat(enable_automatic_function_calling=True)


def _retry_after(exc: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc)) or re.search(r"seconds:\s*(\d+)", str(exc))
    return min(float(match.group(1)) + 1, 30) if match else 10.0


def send_message(chat, prompt: str) -> str:
    """Send a turn, backing off once or twice on a 429 (free-tier rate limit)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat.send_message(prompt).text
        except Exception as exc:
            text = str(exc)
            if "429" in text and attempt < MAX_RETRIES:
                wait = _retry_after(exc)
                with st.spinner(f"Límite gratuito alcanzado, reintento en {wait:.0f} s…"):
                    time.sleep(wait)
                continue
            if "404" in text or "not found" in text.lower():
                models = available_models()
                hint = ("\n\nModelos disponibles: " + ", ".join(models)) if models else ""
                return f"Error: {exc}{hint}"
            if "429" in text:
                return (
                    "Se ha agotado la cuota gratuita de Gemini para este minuto. "
                    "Espera un poco y vuelve a intentarlo, o elige un modelo `-lite` "
                    "en la barra lateral."
                )
            return f"Error al hablar con Gemini: {exc}"
    return "No se pudo completar la petición."


st.title("🏠 Tasador conversacional de pisos")
st.caption(
    f"Modelo entrenado con datos de Fotocasa · conversación con Gemini · "
    f"error medio ≈ {MODEL['meta'].get('mae', 0):,.0f} € · estimación aproximada"
)

api_key = _from_secret_or_env("GEMINI_API_KEY")
if not api_key:
    st.error("Falta `GEMINI_API_KEY` (secret de la app o variable de entorno).")
    st.stop()
genai.configure(api_key=api_key)

preferred = _from_secret_or_env("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
options = available_models() or [preferred]
if preferred not in options:
    options = [preferred, *options]
with st.sidebar:
    model_name = st.selectbox("Modelo Gemini", options, index=options.index(preferred))
    st.caption(
        "El tier gratuito limita las peticiones por minuto; si ves un error 429, "
        "espera unos segundos o elige un modelo `-lite`."
    )

# (Re)build the chat when the app starts or the model changes.
if st.session_state.get("model_name") != model_name:
    try:
        st.session_state.chat = build_chat(model_name)
        st.session_state.model_name = model_name
    except Exception as exc:
        st.error(f"No se pudo cargar `{model_name}`: {exc}")
        st.stop()
chat = st.session_state.chat

for message in chat.history:
    role = "assistant" if message.role == "model" else "user"
    text = "".join(getattr(part, "text", "") for part in message.parts)
    if text.strip():
        with st.chat_message(role):
            st.write(text)

if prompt := st.chat_input("Cuéntame sobre el piso…"):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            answer = send_message(chat, prompt)
        st.write(answer)
