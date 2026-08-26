"""Episode-level logging. Pure storage — never call numpy RNG here."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Optional
import csv
import os


@dataclass
class EpisodeRow:
    seed_set: str          # "A" | "B"
    seed_id: int           # 0..19 (or global seed index)
    init: str              # "misspec" | "aligned"
    arm: str               # "static" | "skip" | "cfeg" | ...
    episode: int           # 0..T-1
    terminal_action: str   # "act_0".."act_5" | "abstain" | "gather" | "ask" | "other"
    success: Optional[int]  # 1/0 if Act, else None
    harm: Optional[float]  # None if not Act
    catastrophic: int      # 1 if harm is not None and harm >= 0.5 else 0
    reward: float
    spend: float
    spend_eff: float
    abstain_flag: int      # 1 if terminal abstain
    n_updates: int         # 0 or 1 this episode (adapter step applied)
    w2_before: float
    w2_after: float
    # Optional but useful for threshold mechanism later:
    u_max: Optional[float] = None   # max available U under current w
    a_top: Optional[int] = None     # argmax tool index or None


def episode_row_to_dict(row: EpisodeRow) -> dict[str, Any]:
    d = asdict(row)
    # CSV-friendly nulls
    for k in ("success", "harm", "u_max", "a_top"):
        if d[k] is None:
            d[k] = ""
    return d


def write_episodes_csv(path: str, rows: list[EpisodeRow]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(episode_row_to_dict(rows[0]).keys())
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(episode_row_to_dict(r))
