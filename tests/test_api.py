"""Stage 3: API tests using FastAPI TestClient.

Tests cover:
1.  GET /health — successful response
2.  POST /predict — valid input with actual model (integration)
3.  POST /predict — other_atms_1km > 30 → 422
4.  POST /predict — avg_ticket_rs < 250 → 422
5.  POST /predict — invalid ATM category → 422
6.  POST /predict — invalid area category → 422
7.  POST /predict — missing required field → 422
8.  POST /predict — malformed JSON → 422
9.  Model loading — model absent → 503
10. Prediction error handling — corrupt model → 500
"""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "nearby_shops": 12,
    "days_to_salary": 7,
    "footfall_index": 55.0,
    "distance_to_market_km": 2.5,
    "avg_ticket_rs": 1500.0,
    "other_atms_1km": 3,
    "atm_type": "mall",
    "area": "Market",
}


def _fresh_app():
    """Return a TestClient with the api module freshly imported.

    Reloading ensures that the module-level ``_model`` cache is reset
    between tests that patch the model.
    """
    import src.mp293.api as api_module

    # Reset cached model so each test starts clean
    api_module._model = None
    api_module._preprocessor = None
    return TestClient(api_module.app)


# ---------------------------------------------------------------------------
# Test 1: GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        client = _fresh_app()
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "model_path" in body


# ---------------------------------------------------------------------------
# Test 2: Valid prediction (integration — requires models/model.joblib)
# ---------------------------------------------------------------------------

class TestPredictValid:
    def test_valid_prediction_with_actual_model(self):
        """Integration test: requires models/model.joblib to exist."""
        import src.mp293.api as api_module
        model_path = api_module.MODEL_PATH
        if not model_path.exists():
            pytest.skip(
                f"models/model.joblib not found at {model_path}. "
                "Run: python -m src.mp293.export_model"
            )
        client = _fresh_app()
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert "predicted_withdrawal_thousand_rs" in body
        assert isinstance(body["predicted_withdrawal_thousand_rs"], float)
        assert body["status"] == "success"
        assert body["model"] == "atm_cash_demand_model@champion"

    def test_valid_prediction_with_mock_model(self):
        """Unit test: uses a mock model so no artifact is needed."""
        import src.mp293.api as api_module

        mock_model = MagicMock()
        mock_model.predict.return_value = [42.5]
        api_module._model = mock_model
        api_module._preprocessor = None

        client = TestClient(api_module.app)
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_withdrawal_thousand_rs"] == pytest.approx(42.5)

        # Reset
        api_module._model = None


# ---------------------------------------------------------------------------
# Test 3: other_atms_1km > 30
# ---------------------------------------------------------------------------

class TestOtherAtmsValidation:
    def test_other_atms_above_30_rejected(self):
        client = _fresh_app()
        payload = {**VALID_PAYLOAD, "other_atms_1km": 31}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_other_atms_exactly_30_accepted_with_mock(self):
        import src.mp293.api as api_module
        mock_model = MagicMock()
        mock_model.predict.return_value = [10.0]
        api_module._model = mock_model
        client = TestClient(api_module.app)
        payload = {**VALID_PAYLOAD, "other_atms_1km": 30}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        api_module._model = None


# ---------------------------------------------------------------------------
# Test 4: avg_ticket_rs < 250
# ---------------------------------------------------------------------------

class TestAvgTicketValidation:
    def test_avg_ticket_below_250_rejected(self):
        client = _fresh_app()
        payload = {**VALID_PAYLOAD, "avg_ticket_rs": 249}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_avg_ticket_exactly_250_accepted_with_mock(self):
        import src.mp293.api as api_module
        mock_model = MagicMock()
        mock_model.predict.return_value = [15.0]
        api_module._model = mock_model
        client = TestClient(api_module.app)
        payload = {**VALID_PAYLOAD, "avg_ticket_rs": 250}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        api_module._model = None


# ---------------------------------------------------------------------------
# Test 5: Invalid ATM category
# ---------------------------------------------------------------------------

class TestAtmTypeValidation:
    @pytest.mark.parametrize("bad_type", ["kiosk", "mobile", "", "MALL", "Bank"])
    def test_invalid_atm_type_rejected(self, bad_type):
        client = _fresh_app()
        payload = {**VALID_PAYLOAD, "atm_type": bad_type}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("good_type", ["bank_branch", "mall", "standalone"])
    def test_valid_atm_types_accepted_with_mock(self, good_type):
        import src.mp293.api as api_module
        mock_model = MagicMock()
        mock_model.predict.return_value = [20.0]
        api_module._model = mock_model
        client = TestClient(api_module.app)
        payload = {**VALID_PAYLOAD, "atm_type": good_type}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        api_module._model = None


# ---------------------------------------------------------------------------
# Test 6: Invalid area category
# ---------------------------------------------------------------------------

class TestAreaValidation:
    @pytest.mark.parametrize("bad_area", ["Rural", "Urban", "highway", "market", ""])
    def test_invalid_area_rejected(self, bad_area):
        client = _fresh_app()
        payload = {**VALID_PAYLOAD, "area": bad_area}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("good_area", ["Highway", "Market", "Residential"])
    def test_valid_areas_accepted_with_mock(self, good_area):
        import src.mp293.api as api_module
        mock_model = MagicMock()
        mock_model.predict.return_value = [18.0]
        api_module._model = mock_model
        client = TestClient(api_module.app)
        payload = {**VALID_PAYLOAD, "area": good_area}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        api_module._model = None


# ---------------------------------------------------------------------------
# Test 7: Missing required fields
# ---------------------------------------------------------------------------

class TestMissingFields:
    @pytest.mark.parametrize("missing_field", list(VALID_PAYLOAD.keys()))
    def test_missing_field_rejected(self, missing_field):
        client = _fresh_app()
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_body_rejected(self):
        client = _fresh_app()
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 8: Malformed request
# ---------------------------------------------------------------------------

class TestMalformedRequest:
    def test_non_json_body_rejected(self):
        client = _fresh_app()
        resp = client.post(
            "/predict",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_extra_fields_rejected(self):
        """extra='forbid' is set on PredictRequest."""
        client = _fresh_app()
        payload = {**VALID_PAYLOAD, "unexpected_field": 999}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_wrong_type_for_numeric_field(self):
        client = _fresh_app()
        payload = {**VALID_PAYLOAD, "nearby_shops": "not-a-number"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 9: Model loading — model file absent
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_missing_model_returns_503(self, tmp_path, monkeypatch):
        import src.mp293.api as api_module

        # Point MODEL_PATH to a non-existent file
        monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.joblib")
        api_module._model = None

        client = TestClient(api_module.app)
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503
        assert "Model artifact not found" in resp.json()["detail"]

        # Restore
        api_module._model = None


# ---------------------------------------------------------------------------
# Test 10: Prediction error handling
# ---------------------------------------------------------------------------

class TestPredictionErrorHandling:
    def test_model_predict_raises_returns_500(self):
        import src.mp293.api as api_module

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("GPU exploded")
        api_module._model = mock_model

        client = TestClient(api_module.app)
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 500
        assert "Prediction failed" in resp.json()["detail"]

        api_module._model = None
