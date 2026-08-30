from keras_neural_network import chat

METADATA = {
    "numeric_features": ["Habitaciones", "Aseos", "Parking", "Colegios"],
    "district_categories": ["Centro", "Retiro", "Salamanca"],
    "feature_medians": {"Habitaciones": 3, "Aseos": 2, "Parking": 0, "Colegios": 9},
}


def test_build_tool_schema_from_metadata():
    tool = chat.build_tool(METADATA)
    props = tool["input_schema"]["properties"]
    assert set(props) == {*METADATA["numeric_features"], "Distrito"}
    assert props["Parking"] == {"type": "integer", "enum": [0, 1]}
    assert props["Habitaciones"]["type"] == "number"
    assert props["Distrito"] == {"type": "string", "enum": METADATA["district_categories"]}
    assert "Distrito" in tool["input_schema"]["required"]


def test_build_system_mentions_features_median_and_district():
    system = chat.build_system(METADATA)
    for name in METADATA["numeric_features"]:
        assert name in system
    assert "median 3" in system
    assert "Distrito" in system


def test_run_tool_success(monkeypatch):
    monkeypatch.setattr(chat, "predict_price", lambda *a, **k: 250000.4)
    out = chat.run_tool(
        ("m", "s", METADATA),
        {"Habitaciones": 3, "Aseos": 2, "Parking": 0, "Colegios": 9, "Distrito": "Retiro"},
    )
    assert out == {"price_eur": 250000}


def test_run_tool_keeps_district_string(monkeypatch):
    seen = {}
    monkeypatch.setattr(chat, "predict_price", lambda m, s, meta, flat: seen.update(flat) or 1.0)
    chat.run_tool(("m", "s", METADATA), {"Habitaciones": 3, "Distrito": "Salamanca"})
    assert seen["Distrito"] == "Salamanca" and seen["Habitaciones"] == 3.0


def test_run_tool_error_is_caught(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad input")

    monkeypatch.setattr(chat, "predict_price", boom)
    out = chat.run_tool(("m", "s", METADATA), {"Habitaciones": 3})
    assert "error" in out and "bad input" in out["error"]
