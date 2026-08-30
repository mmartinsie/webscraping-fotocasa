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

import google.generativeai as genai
import streamlit as st

from pricing import load_model, predict_price

# gemini-1.5-flash is in the free tier; swap for gemini-2.0-flash if you prefer.
GEMINI_MODEL = "gemini-1.5-flash"

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


def get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # no secrets.toml at all
        pass
    return os.environ.get("GEMINI_API_KEY")


st.title("🏠 Tasador conversacional de pisos")
st.caption(
    f"Modelo entrenado con datos de Fotocasa · conversación con Gemini · "
    f"error medio ≈ {MODEL['meta'].get('mae', 0):,.0f} € · estimación aproximada"
)

api_key = get_api_key()
if not api_key:
    st.error("Falta `GEMINI_API_KEY` (secret de la app o variable de entorno).")
    st.stop()
genai.configure(api_key=api_key)

if "chat" not in st.session_state:
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM, tools=[estimar_precio])
    st.session_state.chat = model.start_chat(enable_automatic_function_calling=True)
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
            try:
                answer = chat.send_message(prompt).text
            except Exception as exc:  # network / quota / API errors
                answer = f"Error al hablar con Gemini: {exc}"
        st.write(answer)
