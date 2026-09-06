"""Uji matriks aturan fuzzy (PRD §17.5) + validasi kontrak rekomendasi. Tanpa torch."""
from app.fuzzy import recommend_green
from app.schemas import validate_recommendation, build_recommendation


def rec(q, w, s, online=True):
    return recommend_green(q, w, s, online, 10.0, 60.0)["recommended_green_s"]


def test_high_queue_gives_long_green():
    assert rec(0.9, 10.0, 0) > 45.0


def test_low_idle_gives_short_green():
    assert rec(0.05, 2.0, 0) < 25.0


def test_medium_level2_gives_long():
    assert rec(0.5, 15.0, 2) > 40.0


def test_low_but_long_wait_gives_medium():
    g = rec(0.05, 50.0, 0)
    assert 25.0 <= g <= 45.0


def test_vision_offline_conservative():
    assert rec(0.9, 5.0, 2, online=False) <= 45.0
    assert rec(0.0, 0.0, 0, online=False) <= 25.0


def test_bounds_always_respected():
    for q in (0.0, 0.5, 1.0):
        for w in (0.0, 30.0, 120.0):
            for s in (0, 1, 2):
                g = rec(q, w, s)
                assert 10.0 <= g <= 60.0, (q, w, s, g)


def test_recommendation_contract_validates():
    approaches = {"north": {"recommended_green_s": 42}}
    m = build_recommendation("SIMPANG_TALUN_01", "ESP32_TRAFFIC_01", approaches,
                             valid_for_ms=60000)
    assert validate_recommendation(m, "SIMPANG_TALUN_01", "ESP32_TRAFFIC_01") == []
    bad = dict(m, controller_id="OTHER")
    assert validate_recommendation(bad, "SIMPANG_TALUN_01", "ESP32_TRAFFIC_01") != []
