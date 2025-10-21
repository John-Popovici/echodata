import numpy as np

from models.tabfms.tabicl_impl import TabICLImpl

def test_fit_predict_roundtrip():
    X = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
    y = np.array(["A", "B", "A"], dtype=object)

    model = TabICLImpl().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape