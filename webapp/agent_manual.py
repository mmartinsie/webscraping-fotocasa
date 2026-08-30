"""The same agent, with the function-calling loop written out by hand.

``streamlit_app.py`` uses ``enable_automatic_function_calling=True`` and lets the
SDK run the loop. This CLI does it explicitly so the mechanism is visible: send a
message, look for ``function_call`` parts, run the matching tool, send the results
back, repeat until the model returns plain text.

    GEMINI_API_KEY=...  python webapp/agent_manual.py "3 rooms 90 m2 in Salamanca, no parking"

The exact way to build a ``function_response`` part depends on the
``google-generativeai`` version; adjust ``_tool_result_part`` if your SDK differs.
"""

from __future__ import annotations

import os
import sys

import google.generativeai as genai

from webapp.tools import TOOLS, VALID_DISTRICTS

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
MAX_STEPS = 6
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

SYSTEM = (
    "You estimate the sale price of flats in Madrid. Tools: estimate_price (one "
    "flat) and compare_districts (every district). Ask for missing inputs, then "
    f"call the right tool. Valid districts: {', '.join(VALID_DISTRICTS)}."
)


def _function_calls(response) -> list:
    parts = response.candidates[0].content.parts
    return [p.function_call for p in parts if getattr(p, "function_call", None) and p.function_call.name]


def _tool_result_part(name: str, result: dict):
    return genai.protos.Part(
        function_response=genai.protos.FunctionResponse(name=name, response={"result": result})
    )


def run(prompt: str) -> None:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    chat = genai.GenerativeModel(MODEL, system_instruction=SYSTEM, tools=TOOLS).start_chat()

    response = chat.send_message(prompt)
    for step in range(MAX_STEPS):
        calls = _function_calls(response)
        if not calls:
            break
        results = []
        for call in calls:
            args = {k: v for k, v in dict(call.args).items()}
            print(f"[step {step}] -> {call.name}({args})")
            output = TOOL_MAP[call.name](**args)
            print(f"[step {step}] <- {output}")
            results.append(_tool_result_part(call.name, output))
        response = chat.send_message(genai.protos.Content(parts=results))

    print("\n" + response.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run(" ".join(sys.argv[1:]))
