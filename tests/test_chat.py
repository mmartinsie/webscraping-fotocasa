import chat

METADATA = {
    "features": ["Habitaciones", "Aseos", "Parking"],
    "feature_medians": {"Habitaciones": 3, "Aseos": 2, "Parking": 0},
}


def test_build_tool_schema_from_features():
    tool = chat.build_tool(METADATA["features"])
    props = tool["input_schema"]["properties"]
    assert set(props) == set(METADATA["features"])
    assert props["Parking"] == {"type": "integer", "enum": [0, 1]}
    assert props["Habitaciones"]["type"] == "number"
    assert tool["input_schema"]["required"] == METADATA["features"]


def test_build_system_mentions_every_feature_and_median():
    system = chat.build_system(METADATA)
    for name in METADATA["features"]:
        assert name in system
    assert "median 3" in system


def test_run_tool_success(monkeypatch):
    monkeypatch.setattr(chat, "predict_price", lambda *a, **k: 250000.4)
    out = chat.run_tool(("m", "s", METADATA), {"Habitaciones": 3, "Aseos": 2, "Parking": 0})
    assert out == {"price_eur": 250000}


def test_run_tool_error_is_caught(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad input")

    monkeypatch.setattr(chat, "predict_price", boom)
    out = chat.run_tool(("m", "s", METADATA), {"Habitaciones": 3})
    assert "error" in out and "bad input" in out["error"]
