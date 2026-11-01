"""Which kinds of inputs DataGen and BenchGen write, and how many of each.

Misleading inputs always get a fixed share (MISLEADING_SHARE) of the training data and of the benchmark:
inputs where the obvious answer is wrong (misleading names or wording, distractors, false premises).
They are what a fine-tuned model most needs to learn, and what separates it from the base model.
If the spec's edge cases already name such a category (e.g. "Columns with misleading names"), those
categories get the share; otherwise the generic MISLEADING category is added.

Stdlib only, so it can be tested without ScienceClaw.
"""
import re

TYPICAL = "typical cases"
MISLEADING = "misleading inputs (the obvious answer is wrong)"
TYPICAL_SHARE = 0.4
MISLEADING_SHARE = 0.25

_MISLEADING_RX = re.compile(r"mislead|trick|false premise|distract|decoy|confus|deceptive|red herring|trap", re.I)

MISLEADING_HINT = ("Every item in this batch must be MISLEADING: the obvious or surface-level answer is wrong. "
                   "Use misleading names or wording, distracting details, look-alike options or a false premise, "
                   "so a careless reader gets it wrong. The reference reply must still be the correct answer.")


def is_misleading(category: str) -> bool:
    return bool(_MISLEADING_RX.search(str(category or "")))


def plan_categories(edge_cases) -> dict:
    """{category: share of all items}, shares sum to 1. Keeps the spec's order; duplicates in edge_cases
    (DataGen lists weak categories twice after feedback) give that category a double share of its group."""
    edges = [c for c in (edge_cases or []) if c and c != TYPICAL]
    mis = [c for c in edges if is_misleading(c)] or [MISLEADING]
    other = [c for c in edges if not is_misleading(c)]
    shares = {TYPICAL: TYPICAL_SHARE if other else 1 - MISLEADING_SHARE}
    for group, total in ((mis, MISLEADING_SHARE), (other, 1 - TYPICAL_SHARE - MISLEADING_SHARE)):
        for c in group:
            shares[c] = shares.get(c, 0) + total / len(group)
    return shares


def focus_cycle(edge_cases, length: int = 20) -> list:
    """A repeating list of batch focuses for DataGen whose mix follows plan_categories()."""
    shares = plan_categories(edge_cases)
    counts = {c: max(1, round(s * length)) for c, s in shares.items()}
    out, left = [], dict(counts)
    while any(left.values()):           # interleave, so a short run still sees every category early
        for c in counts:
            if left[c]:
                out.append(c)
                left[c] -= 1
    return out


def focus_hint(category: str) -> str:
    """Extra instruction for a batch of this category ('' for ordinary ones)."""
    return MISLEADING_HINT if is_misleading(category) else ""
