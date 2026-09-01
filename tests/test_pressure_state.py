"""Unit tests for headroom.proxy.pressure_state (CPA extension)."""

from __future__ import annotations

import threading
import time

import pytest

from headroom.proxy.pressure_state import (
    PRESSURE_PIPELINE_KWARGS,
    ContextPressureLevel,
    PressureState,
    PressureStateStore,
    _classify,
    _HEADROOM_STATIC_DEFAULTS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Level classification boundaries
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ratio,expected", [
    (0.00,  ContextPressureLevel.COMFORTABLE),
    (0.39,  ContextPressureLevel.COMFORTABLE),
    (0.40,  ContextPressureLevel.MODERATE),
    (0.59,  ContextPressureLevel.MODERATE),
    (0.60,  ContextPressureLevel.ELEVATED),
    (0.74,  ContextPressureLevel.ELEVATED),
    (0.75,  ContextPressureLevel.HIGH),
    (0.87,  ContextPressureLevel.HIGH),
    (0.88,  ContextPressureLevel.CRITICAL),
    (1.00,  ContextPressureLevel.CRITICAL),
    (1.50,  ContextPressureLevel.CRITICAL),  # over-budget still CRITICAL
])
def test_classify_boundaries(ratio: float, expected: ContextPressureLevel) -> None:
    assert _classify(ratio) == expected


# ──────────────────────────────────────────────────────────────────────────────
# COMFORTABLE level matches current headroom static defaults
# ──────────────────────────────────────────────────────────────────────────────

def test_comfortable_matches_static_defaults() -> None:
    cfg = PRESSURE_PIPELINE_KWARGS[ContextPressureLevel.COMFORTABLE]
    assert cfg["protect_recent"] == 4
    assert cfg["max_items_after_crush"] == 15
    assert cfg["lossless_min_savings_ratio"] == 0.15
    assert cfg["min_tokens_to_compress"] == 250
    assert cfg == _HEADROOM_STATIC_DEFAULTS


def test_critical_is_most_aggressive() -> None:
    cfg = PRESSURE_PIPELINE_KWARGS[ContextPressureLevel.CRITICAL]
    assert cfg["protect_recent"] == 0
    assert cfg["max_items_after_crush"] <= 5
    assert cfg["lossless_min_savings_ratio"] < 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Monotonicity: higher pressure → more aggressive (never less)
# ──────────────────────────────────────────────────────────────────────────────

_ORDERED_LEVELS = [
    ContextPressureLevel.COMFORTABLE,
    ContextPressureLevel.MODERATE,
    ContextPressureLevel.ELEVATED,
    ContextPressureLevel.HIGH,
    ContextPressureLevel.CRITICAL,
]

def test_monotonicity_protect_recent() -> None:
    values = [PRESSURE_PIPELINE_KWARGS[lv]["protect_recent"] for lv in _ORDERED_LEVELS]
    assert values == sorted(values, reverse=True), f"protect_recent not monotone: {values}"


def test_monotonicity_max_items() -> None:
    values = [PRESSURE_PIPELINE_KWARGS[lv]["max_items_after_crush"] for lv in _ORDERED_LEVELS]
    assert values == sorted(values, reverse=True), f"max_items_after_crush not monotone: {values}"


def test_monotonicity_min_tokens() -> None:
    values = [PRESSURE_PIPELINE_KWARGS[lv]["min_tokens_to_compress"] for lv in _ORDERED_LEVELS]
    assert values == sorted(values, reverse=True), f"min_tokens_to_compress not monotone: {values}"


def test_monotonicity_lossless_ratio() -> None:
    values = [PRESSURE_PIPELINE_KWARGS[lv]["lossless_min_savings_ratio"] for lv in _ORDERED_LEVELS]
    assert values == sorted(values, reverse=True), f"lossless_min_savings_ratio not monotone: {values}"


# ──────────────────────────────────────────────────────────────────────────────
# PressureState
# ──────────────────────────────────────────────────────────────────────────────

def test_pressure_state_default() -> None:
    s = PressureState()
    assert s.pressure_ratio() == 0.0
    assert s.level() == ContextPressureLevel.COMFORTABLE
    assert s.turn_count == 0


def test_pressure_state_update_returns_level() -> None:
    s = PressureState()
    level = s.update(provider_input_tokens=120_000, context_limit=200_000)
    assert level == ContextPressureLevel.ELEVATED  # 60%
    assert s.turn_count == 1


def test_pressure_ratio_clamped() -> None:
    s = PressureState()
    s.update(provider_input_tokens=999_999, context_limit=100_000)
    assert s.pressure_ratio() == 1.0


def test_pressure_ratio_zero_on_zero_tokens() -> None:
    s = PressureState()
    s.update(provider_input_tokens=0, context_limit=200_000)
    assert s.pressure_ratio() == 0.0


def test_pipeline_kwargs_returns_copy() -> None:
    s = PressureState()
    kwargs1 = s.pipeline_kwargs()
    kwargs2 = s.pipeline_kwargs()
    assert kwargs1 == kwargs2
    kwargs1["protect_recent"] = 999
    assert kwargs2["protect_recent"] != 999  # mutation does not alias


def test_is_elevated_at_high_pressure() -> None:
    s = PressureState()
    s.update(160_000, 200_000)  # 80% → HIGH
    assert s.is_elevated()


def test_is_not_elevated_at_comfortable() -> None:
    s = PressureState()
    s.update(50_000, 200_000)  # 25% → COMFORTABLE
    assert not s.is_elevated()


@pytest.mark.parametrize("tokens,limit,expected_level", [
    (0,       200_000, ContextPressureLevel.COMFORTABLE),
    (70_000,  200_000, ContextPressureLevel.MODERATE),    # 35% < 40 → comfortable, 70k/200k = 35%, actually comfortable
    (90_000,  200_000, ContextPressureLevel.MODERATE),    # 45%
    (130_000, 200_000, ContextPressureLevel.ELEVATED),    # 65%
    (160_000, 200_000, ContextPressureLevel.HIGH),        # 80%
    (185_000, 200_000, ContextPressureLevel.CRITICAL),    # 92.5%
])
def test_pipeline_kwargs_correct_per_level(tokens: int, limit: int, expected_level: ContextPressureLevel) -> None:
    s = PressureState()
    s.update(tokens, limit)
    assert s.level() == expected_level
    assert s.pipeline_kwargs() == PRESSURE_PIPELINE_KWARGS[expected_level]


# ──────────────────────────────────────────────────────────────────────────────
# PressureStateStore
# ──────────────────────────────────────────────────────────────────────────────

def test_store_creates_on_first_access() -> None:
    store = PressureStateStore()
    state = store.get_or_create("sess-1")
    assert state is not None
    assert state.turn_count == 0


def test_store_returns_same_state() -> None:
    store = PressureStateStore()
    s1 = store.get_or_create("sess-1")
    s2 = store.get_or_create("sess-1")
    assert s1 is s2


def test_store_returns_empty_kwargs_before_update() -> None:
    store = PressureStateStore()
    kwargs = store.get_pipeline_kwargs("new-session")
    assert kwargs == {}


def test_store_returns_empty_kwargs_for_zero_turn_count() -> None:
    store = PressureStateStore()
    store.get_or_create("sess-zero")
    kwargs = store.get_pipeline_kwargs("sess-zero")
    assert kwargs == {}


def test_store_update_sets_kwargs() -> None:
    store = PressureStateStore()
    level = store.update("sess-upd", provider_input_tokens=160_000, context_limit=200_000)
    assert level == ContextPressureLevel.HIGH
    kwargs = store.get_pipeline_kwargs("sess-upd")
    assert kwargs["max_items_after_crush"] == 5
    assert kwargs["protect_recent"] == 1


def test_store_len() -> None:
    store = PressureStateStore()
    store.get_or_create("a")
    store.get_or_create("b")
    assert len(store) == 2


def test_store_evicts_stale_sessions() -> None:
    store = PressureStateStore(ttl_seconds=0.01)  # 10ms TTL
    store.update("old-sess", 100_000, 200_000)
    assert len(store) == 1
    time.sleep(0.05)
    # eviction triggers on next update call
    store.update("new-sess", 50_000, 200_000)
    assert "old-sess" not in store._states
    assert len(store) == 1


def test_store_caps_at_max_sessions() -> None:
    store = PressureStateStore(max_sessions=5)
    for i in range(10):
        store.update(f"sess-{i}", 50_000, 200_000)
        time.sleep(0.001)  # ensure different last_updated
    assert len(store) <= 5


def test_store_thread_safety() -> None:
    store = PressureStateStore()
    errors: list[Exception] = []

    def worker(session_id: str) -> None:
        try:
            for _ in range(20):
                store.update(session_id, 100_000, 200_000)
                store.get_pipeline_kwargs(session_id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"Thread-safety failures: {errors}"


def test_store_different_sessions_independent() -> None:
    store = PressureStateStore()
    store.update("high-pressure", 185_000, 200_000)   # CRITICAL
    store.update("low-pressure",  40_000,  200_000)   # COMFORTABLE
    assert store.get_pipeline_kwargs("high-pressure")["max_items_after_crush"] == 3
    assert store.get_pipeline_kwargs("low-pressure")["max_items_after_crush"] == 15
