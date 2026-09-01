"""Context Pressure-Aware Adaptive Compression (CPA).

Tracks the fraction of a model's context window consumed per session and
returns escalating SmartCrusher aggressiveness kwargs for pipeline.apply().

The feedback loop is intentionally one-turn lagged: pressure is observed
from the provider's usage response at turn N and applied to the compression
config at turn N+1.  This is the correct causal ordering — the proxy cannot
know the true token count before the provider confirms it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Pressure levels
# ──────────────────────────────────────────────────────────────────────────────

class ContextPressureLevel(str, Enum):
    """Bucketed context-window utilisation levels."""
    COMFORTABLE = "comfortable"   # < 40 %
    MODERATE    = "moderate"      # 40 – 60 %
    ELEVATED    = "elevated"      # 60 – 75 %
    HIGH        = "high"          # 75 – 88 %
    CRITICAL    = "critical"      # ≥ 88 %


def _classify(ratio: float) -> ContextPressureLevel:
    if ratio < 0.40:
        return ContextPressureLevel.COMFORTABLE
    if ratio < 0.60:
        return ContextPressureLevel.MODERATE
    if ratio < 0.75:
        return ContextPressureLevel.ELEVATED
    if ratio < 0.88:
        return ContextPressureLevel.HIGH
    return ContextPressureLevel.CRITICAL


# ──────────────────────────────────────────────────────────────────────────────
# Per-level pipeline kwargs
# ──────────────────────────────────────────────────────────────────────────────
#
# Each entry maps directly to kwargs accepted by TransformPipeline.apply() and
# forwarded to ContentRouter / SmartCrusher.  All keys are recognised by
# proxy_pipeline_kwargs() consumers (content_router.py, smart_crusher.py).
#
# Invariant: for any two levels L1 < L2 (more pressure):
#   protect_recent(L1) >= protect_recent(L2)
#   max_items_after_crush(L1) >= max_items_after_crush(L2)
#   min_tokens_to_compress(L1) >= min_tokens_to_compress(L2)
# i.e. aggressiveness is monotone in pressure (proven by test_monotonicity).

PRESSURE_PIPELINE_KWARGS: dict[ContextPressureLevel, dict[str, Any]] = {
    ContextPressureLevel.COMFORTABLE: {
        "protect_recent":             4,
        "max_items_after_crush":      15,
        "lossless_min_savings_ratio": 0.15,
        "min_tokens_to_compress":     250,
    },
    ContextPressureLevel.MODERATE: {
        "protect_recent":             3,
        "max_items_after_crush":      12,
        "lossless_min_savings_ratio": 0.12,
        "min_tokens_to_compress":     200,
    },
    ContextPressureLevel.ELEVATED: {
        "protect_recent":             2,
        "max_items_after_crush":      8,
        "lossless_min_savings_ratio": 0.08,
        "min_tokens_to_compress":     150,
    },
    ContextPressureLevel.HIGH: {
        "protect_recent":             1,
        "max_items_after_crush":      5,
        "lossless_min_savings_ratio": 0.04,
        "min_tokens_to_compress":     100,
    },
    ContextPressureLevel.CRITICAL: {
        "protect_recent":             0,
        "max_items_after_crush":      3,
        "lossless_min_savings_ratio": 0.01,
        "min_tokens_to_compress":     50,
    },
}

# Values at COMFORTABLE match headroom's current static defaults so CPA is a
# strict no-op at low pressure and only kicks in as the window fills.
_HEADROOM_STATIC_DEFAULTS = PRESSURE_PIPELINE_KWARGS[ContextPressureLevel.COMFORTABLE]


# ──────────────────────────────────────────────────────────────────────────────
# Per-session state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PressureState:
    """Mutable per-session context-pressure state."""

    last_provider_input_tokens: int = 0
    last_context_limit: int = 200_000
    turn_count: int = 0
    last_updated: float = field(default_factory=time.monotonic)

    def update(self, provider_input_tokens: int, context_limit: int) -> ContextPressureLevel:
        """Record observed token count; return new pressure level."""
        self.last_provider_input_tokens = max(0, provider_input_tokens)
        self.last_context_limit = max(1, context_limit)
        self.turn_count += 1
        self.last_updated = time.monotonic()
        return self.level()

    def pressure_ratio(self) -> float:
        """Fraction of context window consumed, clamped to [0.0, 1.0]."""
        return min(1.0, max(0.0, self.last_provider_input_tokens / self.last_context_limit))

    def level(self) -> ContextPressureLevel:
        return _classify(self.pressure_ratio())

    def pipeline_kwargs(self) -> dict[str, Any]:
        """Return compression kwargs appropriate for the current pressure level."""
        return dict(PRESSURE_PIPELINE_KWARGS[self.level()])

    def is_elevated(self) -> bool:
        return self.level() not in (
            ContextPressureLevel.COMFORTABLE,
            ContextPressureLevel.MODERATE,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Session store
# ──────────────────────────────────────────────────────────────────────────────

_SESSION_TTL_SECONDS = 3600          # evict sessions idle > 1 hour
_MAX_SESSIONS = 10_000               # memory cap


class PressureStateStore:
    """Thread-safe mapping of session_id → PressureState."""

    def __init__(
        self,
        ttl_seconds: float = _SESSION_TTL_SECONDS,
        max_sessions: int = _MAX_SESSIONS,
    ) -> None:
        self._states: dict[str, PressureState] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max = max_sessions

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> PressureState:
        with self._lock:
            if session_id not in self._states:
                self._states[session_id] = PressureState()
            return self._states[session_id]

    def update(
        self,
        session_id: str,
        provider_input_tokens: int,
        context_limit: int,
    ) -> ContextPressureLevel:
        """Update state for *session_id* and return the new pressure level."""
        with self._lock:
            if session_id not in self._states:
                self._states[session_id] = PressureState()
            level = self._states[session_id].update(provider_input_tokens, context_limit)
            self._evict_old_sessions()
            return level

    def get_pipeline_kwargs(self, session_id: str) -> dict[str, Any]:
        """Return compression kwargs for the session.

        Returns an empty dict (no overrides) on the first turn — before any
        response has been observed — so the pipeline uses its static defaults.
        This is the correct conservative behaviour: never increase aggressiveness
        on a turn whose pressure is unknown.
        """
        with self._lock:
            state = self._states.get(session_id)
        if state is None or state.turn_count == 0:
            return {}
        return state.pipeline_kwargs()

    # ── Maintenance ───────────────────────────────────────────────────────────

    def _evict_old_sessions(self) -> None:
        """Remove sessions idle longer than TTL or when store exceeds max size.

        Called under self._lock.
        """
        now = time.monotonic()
        cutoff = now - self._ttl
        stale = [sid for sid, s in self._states.items() if s.last_updated < cutoff]
        for sid in stale:
            del self._states[sid]
        # If still over capacity, remove oldest-updated sessions
        if len(self._states) > self._max:
            by_age = sorted(self._states.items(), key=lambda kv: kv[1].last_updated)
            for sid, _ in by_age[: len(self._states) - self._max]:
                del self._states[sid]

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)
