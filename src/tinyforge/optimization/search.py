from dataclasses import dataclass, asdict
import itertools
import random
from typing import Any, Callable, Iterable

@dataclass(frozen=True)
class SearchResult:
    params: dict[str, Any]
    score: float
    trials: list[dict[str, Any]]
    def to_dict(self): return asdict(self)

def _run(candidates: Iterable[dict[str, Any]], objective: Callable[[dict[str, Any]], float],
         maximize: bool = True) -> SearchResult:
    trials = []
    for params in candidates:
        score = float(objective(params))
        trials.append({"params": dict(params), "score": score})
    if not trials: raise ValueError("search space is empty")
    key = lambda x: (x["score"], tuple(sorted(map(str, x["params"].items()))))
    best = (max if maximize else min)(trials, key=key)
    return SearchResult(best["params"], best["score"], trials)

def grid_search(space: dict[str, Iterable[Any]], objective: Callable[[dict[str, Any]], float],
                *, maximize: bool = True) -> SearchResult:
    keys = list(space)
    return _run((dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))),
                objective, maximize)

def random_search(space: dict[str, Iterable[Any]], objective: Callable[[dict[str, Any]], float],
                  *, trials: int = 10, seed: int = 0, maximize: bool = True) -> SearchResult:
    if trials < 1: raise ValueError("trials must be positive")
    options = {k: tuple(v) for k, v in space.items()}
    if any(not v for v in options.values()): raise ValueError("search space is empty")
    rng = random.Random(seed)
    return _run(({k: rng.choice(v) for k, v in options.items()} for _ in range(trials)),
                objective, maximize)
