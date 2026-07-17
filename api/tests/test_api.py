from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_registry_shape():
    r = client.get("/api/registry").json()
    assert {"models", "modules", "live_count", "total"} <= set(r)
    assert "identify" in r["modules"]


def test_classify_stub_without_model(monkeypatch):
    # Force "no model loaded" so the deterministic stub path is exercised.
    from api.ml import classifier

    monkeypatch.setattr(classifier, "_load", lambda: None)
    r = client.post("/api/classify", data={"sample": "coneflower"}).json()
    assert len(r["predictions"]) == 5
    assert r["predictions"][0]["family"] == "Asteraceae"
    assert r["served"] == "stub"
