"""Plan batches so a skill call finishes inside the executor's time limit.

Model: one generate+verify round for k items takes about 2*CALL_OVERHEAD_S + per_ex*k seconds;
per_ex is learned per run. SAFETY pads every estimate."""
import time

CALL_OVERHEAD_S = 2.5     # fixed latency per LLM call, on top of time that scales with batch size
SAFETY = 1.3              # plan rounds as if they take 30% longer than estimated


def est_round(k: int, per_ex: float, calls: int = 2) -> float:
    """Planned duration of one round (`calls` LLM calls) for k items, including the safety factor."""
    return SAFETY * (calls * CALL_OVERHEAD_S + per_ex * k)

def plan_batch(remaining: float, per_ex: float, max_k: int, calls: int = 2) -> int:
    """Largest k <= max_k whose planned round fits in `remaining` seconds (0 if none)."""
    if remaining == float("inf"):
        return max(0, max_k)
    k = int((remaining / SAFETY - calls * CALL_OVERHEAD_S) / max(per_ex, 1e-6))
    return max(0, min(k, max_k))

def update_per_ex(per_ex: float, took: float, k: int, calls: int = 2) -> float:
    """Blend in a new observation. Bigger rounds are weighted more: a 1-example round is mostly
    fixed overhead and says little about per-example cost."""
    obs = max(0.2, (took - calls * CALL_OVERHEAD_S) / max(k, 1))
    alpha = min(0.6, k / 15)
    return (1 - alpha) * per_ex + alpha * obs

def regen_fits(need: int, per_ex: float, deadline: float, calls: int = 2) -> bool:
    return time.time() + est_round(need, per_ex, calls) <= deadline

