"""Conversational price estimator.

Claude chats with you, collects the flat's five features and calls the saved
model to give you a recommended price.

    python recommend_price.py --save model_dir     # once: train + save the model
    export ANTHROPIC_API_KEY=sk-ant-...            # or: ant auth login
    python chat.py model_dir

Model defaults to claude-opus-5; pass --model claude-haiku-4-5 for a cheaper run.
"""

from __future__ import annotations

import argparse
import json
import os

import anthropic

from predict import load_bundle, predict_price

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# The tool feeds these dataset column names; keep them in sync with dataset.FEATURES.
FEATURE_MAP = {
    "habitaciones": "Habitaciones",
    "aseos": "Aseos",
    "superficie": "Superficie",
    "parking": "Parking",
    "colegios": "Colegios",
}

SYSTEM = """You help a user estimate the sale price of a flat in Madrid.

Converse in the user's language (Spanish by default). Keep replies short.

Before you can give a price you need these five values:
- habitaciones: number of rooms (integer)
- aseos: number of bathrooms (integer)
- superficie: floor area in m2 (number)
- parking: 1 if the flat has parking, else 0
- colegios: number of schools nearby. If the user does not know, use 9
  (the dataset median) and say so.

Ask for whatever is missing, a couple of items at a time, and accept values
given in any order or all at once. When you have all five, briefly echo them
back, then call the predict_price tool exactly once. Report the returned figure
in euros and make clear it is a rough model estimate, not a professional
appraisal."""

TOOL = {
    "name": "predict_price",
    "description": "Estimate the flat's sale price. Call once all five features are known.",
    "input_schema": {
        "type": "object",
        "properties": {
            "habitaciones": {"type": "integer", "description": "number of rooms"},
            "aseos": {"type": "integer", "description": "number of bathrooms"},
            "superficie": {"type": "number", "description": "floor area in m2"},
            "parking": {"type": "integer", "enum": [0, 1]},
            "colegios": {"type": "number", "description": "schools nearby (default 9)"},
        },
        "required": ["habitaciones", "aseos", "superficie", "parking", "colegios"],
        "additionalProperties": False,
    },
}


def run_tool(bundle: tuple, tool_input: dict) -> dict:
    model, scaler, metadata = bundle
    flat = {FEATURE_MAP[k]: v for k, v in tool_input.items() if k in FEATURE_MAP}
    price = predict_price(model, scaler, metadata, flat)
    return {"price_eur": round(price), "features": flat}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", help="bundle dir from recommend_price.py --save")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.model_dir):
        raise SystemExit(
            f"No model bundle at {args.model_dir!r}. "
            f"Create one with:  python recommend_price.py --save {args.model_dir}"
        )

    bundle = load_bundle(args.model_dir)
    client = anthropic.Anthropic()
    messages: list[dict] = []

    print("Chat de tasación de pisos (Ctrl-D o 'salir' para terminar).\n")
    while True:
        try:
            user = input("tú> ").strip()
        except EOFError:
            print()
            break
        if user.lower() in {"salir", "exit", "quit"}:
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})

        # Inner loop: keep going while Claude wants to call the tool.
        while True:
            response = client.messages.create(
                model=args.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                tools=[TOOL],
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\nClaude> {block.text.strip()}\n")

            if response.stop_reason == "refusal":
                print("\nClaude> (no puedo continuar con esta petición)\n")
                break
            if response.stop_reason != "tool_use":
                break

            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(run_tool(bundle, block.input)),
                }
                for block in response.content
                if block.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
