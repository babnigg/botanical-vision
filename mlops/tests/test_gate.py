import mlflow.tracking

from mlops import promote as P


class _FakeMV:
    version = "1"


def _fake_client(set_calls):
    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            return _FakeMV()

        def set_registered_model_alias(self, name, alias, version):
            set_calls.append((alias, version))

    return FakeClient


def _wire(monkeypatch, set_calls, challenger_top1):
    # champion (v1) fixed at 0.5; challenger scores challenger_top1.
    monkeypatch.setattr(P.config, "configure", lambda: None)
    monkeypatch.setattr(
        P, "evaluate",
        lambda uri, limit: {"n": 10, "top5": 1.0,
                            "top1": challenger_top1 if uri.endswith("/2") else 0.5},
    )
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _fake_client(set_calls))


def test_gate_promotes_when_challenger_wins(monkeypatch):
    calls = []
    _wire(monkeypatch, calls, challenger_top1=0.9)
    r = P.promote("2", margin=0.0, limit=5)
    assert r["promoted"] is True
    assert calls == [("production", "2")]


def test_gate_rejects_when_challenger_loses(monkeypatch):
    calls = []
    _wire(monkeypatch, calls, challenger_top1=0.3)
    r = P.promote("2", margin=0.0, limit=5)
    assert r["promoted"] is False
    assert calls == []
