"""LLM-agent demo: a Madrid flat price estimator.

Shows a tool-use / function-calling loop end to end:

    system prompt + tool schema  ->  Gemini asks for the missing inputs
    ->  Gemini calls estimate_price(district, rooms, ...)  (SDK auto-runs it)
    ->  the tool computes the price locally (webapp/pricing.py, NumPy, no TF)
    ->  Gemini turns the result into a sentence

The "Chat" tab drives it with Gemini and shows every tool call/result; the
"Form" tab calls the same tool directly, so the app still works when Gemini's
free-tier quota is exhausted.

Deploy on Streamlit Community Cloud with the main file set to
``webapp/streamlit_app.py`` and a ``GEMINI_API_KEY`` secret (free key at
https://aistudio.google.com/apikey).

Run locally:  GEMINI_API_KEY=...  streamlit run webapp/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import re
import time

import google.generativeai as genai
import streamlit as st

from tools import TOOLS, VALID_DISTRICTS, estimate_price

# gemini-*-lite models get the most generous free-tier request quota, which
# matters here: automatic function calling makes 2-3 API calls per user turn.
# Pin another with a GEMINI_MODEL secret / env var or the sidebar picker.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_RETRIES = 2
RANGE_PCT = 0.15  # +/- band shown around the point estimate

st.set_page_config(page_title="Madrid flat price chat", page_icon="🏠")

SYSTEM = (
    "You help estimate the sale price of flats in Madrid. Reply in the same "
    "language the user writes in (English or Spanish); keep answers short. "
    "Two tools are available: `estimate_price` (one flat: needs district, rooms, "
    "bathrooms, floor area in m2, parking yes/no - nearby schools are derived "
    "from the district) and `compare_districts` (the same flat priced across "
    "every district: needs only area and parking). Pick the right one, ask for "
    "whatever is missing, accept values in any order. After `estimate_price` "
    "report the euro figure and mention the thesis neural-network reference "
    f"(~2020). Valid districts: {', '.join(VALID_DISTRICTS)}. Make clear these "
    "are rough estimates, not an appraisal."
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
        "tab_chat": "Chat",
        "tab_form": "Form",
        "how": "How this agent works",
        "how_body": "A system prompt + two tool schemas (`estimate_price`, "
        "`compare_districts`) are sent to Gemini. It picks a tool, asks for "
        "whatever is missing, then emits a function call; the SDK runs it locally "
        "and feeds the result back, and Gemini phrases the answer. Every "
        "call/result is shown below.",
        "sb_turns": "Turns",
        "sb_tokens": "Tokens",
        "sb_latency": "Last turn",
        "examples": [
            "3 rooms, 2 baths, 90 m², Salamanca, no parking",
            "2-bed 70 m² flat in Carabanchel with parking",
        ],
        "f_district": "District",
        "f_rooms": "Rooms",
        "f_baths": "Bathrooms",
        "f_area": "Floor area (m²)",
        "f_parking": "Has parking",
        "f_go": "Estimate",
        "f_result": "Estimated price",
        "f_nn": "Neural-network reference (~2020)",
        "f_schools": "{n} schools (from the district)",
        "tool_call": "🔧 tool call",
        "tool_result": "↩ tool result",
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
        "tab_chat": "Chat",
        "tab_form": "Formulario",
        "how": "Cómo funciona este agente",
        "how_body": "Se envía a Gemini un system prompt + dos esquemas de "
        "herramienta (`estimate_price`, `compare_districts`). Elige una, pregunta "
        "lo que falte, luego emite una llamada a función; el SDK la ejecuta en "
        "local y le devuelve el resultado, y Gemini redacta la respuesta. Cada "
        "llamada/resultado se muestra abajo.",
        "sb_turns": "Turnos",
        "sb_tokens": "Tokens",
        "sb_latency": "Último turno",
        "examples": [
            "3 habitaciones, 2 baños, 90 m², Salamanca, sin parking",
            "piso de 2 hab, 70 m², en Carabanchel con garaje",
        ],
        "f_district": "Distrito",
        "f_rooms": "Habitaciones",
        "f_baths": "Aseos",
        "f_area": "Superficie (m²)",
        "f_parking": "Tiene parking",
        "f_go": "Estimar",
        "f_result": "Precio estimado",
        "f_nn": "Referencia red neuronal (~2020)",
        "f_schools": "{n} colegios (según el distrito)",
        "tool_call": "🔧 llamada a herramienta",
        "tool_result": "↩ resultado",
    },
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
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM, tools=TOOLS)
    return model.start_chat(enable_automatic_function_calling=True)


def _retry_after(exc: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc)) or re.search(r"seconds:\s*(\d+)", str(exc))
    return min(float(match.group(1)) + 1, 30) if match else 10.0


def price_range(value: float) -> str:
    return f"± {value * RANGE_PCT:,.0f} €"


def _to_dict(mapping) -> dict:
    try:
        return {k: v for k, v in dict(mapping).items()}
    except Exception:
        return {}


def render_history(chat, t: dict) -> None:
    """Replay the conversation, surfacing every function call and its result."""
    for message in chat.history:
        role = "assistant" if message.role == "model" else "user"
        for part in message.parts:
            text = getattr(part, "text", "") or ""
            if text.strip():
                with st.chat_message(role):
                    st.write(text)
            call = getattr(part, "function_call", None)
            if call and getattr(call, "name", ""):
                args = ", ".join(f"{k}={v}" for k, v in _to_dict(call.args).items())
                with st.chat_message("assistant"):
                    st.caption(f"{t['tool_call']}: `{call.name}({args})`")
            resp = getattr(part, "function_response", None)
            if resp and getattr(resp, "name", ""):
                with st.chat_message("assistant"):
                    st.caption(t["tool_result"])
                    st.code(json.dumps(_to_dict(resp.response), ensure_ascii=False, indent=2), "json")


def send_message(chat, prompt: str, t: dict) -> tuple[str, int]:
    """Send a turn (retrying on a 429). Returns ``(text, tokens_used)``."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = chat.send_message(prompt)
            usage = getattr(response, "usage_metadata", None)
            return response.text, getattr(usage, "total_token_count", 0)
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
                return f"Error: {exc}{hint}", 0
            if "429" in text:
                return t["quota"], 0
            return t["gemini_error"].format(exc=exc), 0
    return t["failed"], 0


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

st.session_state.setdefault("turns", 0)
st.session_state.setdefault("tokens", 0)
st.session_state.setdefault("latency", 0.0)

with st.sidebar:
    with st.expander(t["how"]):
        st.markdown(t["how_body"])
    m1, m2, m3 = st.columns(3)
    m1.metric(t["sb_turns"], st.session_state.turns)
    m2.metric(t["sb_tokens"], f"{st.session_state.tokens:,}")
    m3.metric(t["sb_latency"], f"{st.session_state.latency:.1f}s")

tab_chat, tab_form = st.tabs([t["tab_chat"], t["tab_form"]])

# --- Form tab: calls the tool directly, always works ---------------------- #
with tab_form:
    with st.form("estimate"):
        district = st.selectbox(t["f_district"], VALID_DISTRICTS)
        c1, c2, c3 = st.columns(3)
        rooms = c1.number_input(t["f_rooms"], 1, 20, 3)
        baths = c2.number_input(t["f_baths"], 1, 10, 2)
        area = c3.number_input(t["f_area"], 15, 1000, 90)
        parking = st.checkbox(t["f_parking"])
        go = st.form_submit_button(t["f_go"])
    if go:
        r = estimate_price(district, int(rooms), int(baths), float(area), int(parking))
        st.metric(t["f_result"], f"{r['price_eur']:,.0f} €", price_range(r["price_eur"]), delta_color="off")
        st.caption(f"{r['method']} · " + t["f_schools"].format(n=r["schools_by_district"]))
        st.caption(f"{t['f_nn']}: {r['reference_neural_network_2020_eur']:,.0f} €")

# --- Chat tab: the Gemini agent ----------------------------------------- #
prompt = st.chat_input(t["input"])
with tab_chat:
    cols = st.columns(len(t["examples"]))
    for col, example in zip(cols, t["examples"]):
        if col.button(example, use_container_width=True):
            prompt = example

    render_history(chat, t)

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner(t["thinking"]):
            started = time.perf_counter()
            answer, tokens = send_message(chat, prompt, t)
        st.session_state.latency = time.perf_counter() - started
        st.session_state.turns += 1
        st.session_state.tokens += tokens
        with st.chat_message("assistant"):
            st.write(answer)
        st.rerun()  # refresh the history view and the sidebar counters
