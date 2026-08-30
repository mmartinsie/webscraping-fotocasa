"""Conversational price estimator.

Claude chats with you, collects the flat's features and calls the saved model to
give you a recommended price. The questions and the tool are built from the saved
bundle, so a model retrained on different features just works.

    python recommend_price.py --save model_dir     # once: train + save the model
    export ANTHROPIC_API_KEY=sk-ant-...            # or: ant auth login
    python chat.py model_dir

Model defaults to claude-opus-5; pass --model claude-haiku-4-5 for a cheaper run.
"""

from __future__ import annotations

import argparse
import json
import os

from keras_neural_network.predict import BundleError, load_bundle, predict_price

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

SYSTEM_TEMPLATE = """You help a user estimate the sale price of a flat in Madrid.

Converse in the user's language (Spanish by default). Keep replies short.

Before you can give a price you need these values (with the dataset median as a
fallback if the user does not know one):
{feature_lines}

Ask for whatever is missing, a couple of items at a time, and accept values in
any order or all at once. When you have them, briefly echo them back, then call
the predict_price tool exactly once. Report the returned figure in euros and make
clear it is a rough model estimate, not a professional appraisal."""


def _numeric_features(metadata: dict) -> list[str]:
    return metadata.get("numeric_features") or metadata.get("features", [])


def build_tool(metadata: dict) -> dict:
    """A predict_price tool whose schema mirrors the model's inputs."""
    numeric = _numeric_features(metadata)
    properties = {}
    for name in numeric:
        if name.lower() in ("parking", "garaje"):
            properties[name] = {"type": "integer", "enum": [0, 1]}
        else:
            properties[name] = {"type": "number"}
    required = list(numeric)
    categories = metadata.get("district_categories")
    if categories:
        properties["Distrito"] = {"type": "string", "enum": categories}
        required.append("Distrito")
    return {
        "name": "predict_price",
        "description": "Estimate the flat's sale price once the features are known.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def build_system(metadata: dict) -> str:
    medians = metadata.get("feature_medians", {})
    lines = [f"- {name} (median {medians.get(name, 'n/a')})" for name in _numeric_features(metadata)]
    if metadata.get("district_categories"):
        lines.append("- Distrito (one of the Madrid districts)")
    return SYSTEM_TEMPLATE.format(feature_lines="\n".join(lines))


def run_tool(bundle: tuple, tool_input: dict) -> dict:
    try:
        model, scaler, metadata = bundle
        flat = {k: (v if k == "Distrito" else float(v)) for k, v in tool_input.items()}
        return {"price_eur": round(predict_price(model, scaler, metadata, flat))}
    except Exception as exc:  # surface the failure to Claude instead of crashing
        return {"error": f"{type(exc).__name__}: {exc}"}


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
    try:
        bundle = load_bundle(args.model_dir)
    except BundleError as exc:
        raise SystemExit(str(exc)) from exc
    _, _, metadata = bundle

    import anthropic

    tool = build_tool(metadata)
    system = build_system(metadata)
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

        while True:  # keep going while Claude wants to call the tool
            response = client.messages.create(
                model=args.model,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=[tool],
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

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = run_tool(bundle, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                        "is_error": "error" in result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
