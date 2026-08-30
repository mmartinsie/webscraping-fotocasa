import numpy as np

from keras_neural_network.metrics import format_row, score


def test_score_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    m = score(y, y)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["r2"] == 1
    assert m["mape"] == 0


def test_score_known_values():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    m = score(y_true, y_pred)
    assert m["mae"] == 15
    assert round(m["mape"], 1) == 10.0
    assert m["rmse"] > m["mae"]


def test_format_row_has_all_fields():
    line = format_row("demo", score(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert line.startswith("demo")
    assert "%" in line
