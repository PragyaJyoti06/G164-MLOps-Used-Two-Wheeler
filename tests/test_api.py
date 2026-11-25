import math

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


VALID_INPUT = {
    "age_years": 5,
    "km_driven": 25000,
    "engine_cc": 150,
    "owners": 1,
    "service_records": 8,
    "accidents": 0,
    "brand_tier": "mid",
    "city_zone": "North"
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=VALID_INPUT
    )

    assert response.status_code == 200
    assert "prediction" in response.json()

    prediction = response.json()["prediction"]

    assert isinstance(prediction, (int, float))
    assert math.isfinite(prediction)


def test_invalid_age():
    data = VALID_INPUT.copy()
    data["age_years"] = 35

    response = client.post("/predict", json=data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "age_years"]
        for error in response.json()["detail"]
    )


def test_invalid_engine_capacity():
    data = VALID_INPUT.copy()
    data["engine_cc"] = 40

    response = client.post("/predict", json=data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "engine_cc"]
        for error in response.json()["detail"]
    )


def test_missing_field():
    data = VALID_INPUT.copy()
    del data["owners"]

    response = client.post("/predict", json=data)

    assert response.status_code == 422
