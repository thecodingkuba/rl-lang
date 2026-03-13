"""
Fit IRT model parameters (beta, alpha_0, delta) to the Duolingo SLAM dataset
using maximum likelihood estimation.

This is the *supervised learning* component of the project: we calibrate
the simulated learner so it matches real human learning behavior. The RL
routing policies then train on this calibrated environment.

Setup:
  1. Download the SLAM English track from Harvard Dataverse:
     https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO
  2. Place the train file at:  data/en_es.slam.20171218.train
     (or any .train file — pass its path via --data-path)
  3. Run:  python3 -m src.fit_duolingo

Output:
  Prints fitted {beta, alpha_0, delta} and saves them to data/fitted_params.json.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ─── SLAM parsing ────────────────────────────────────────────────────

@dataclass
class Token:
    token_id: str
    word: str
    pos: str
    morph: str
    dep_label: str
    dep_head: int
    label: int  # 0 = correct, 1 = mistake


@dataclass
class Exercise:
    user: str
    days: float
    session: str
    format: str
    time: Optional[float]
    tokens: List[Token]


def parse_slam_file(path: str, max_exercises: int = 200_000) -> List[Exercise]:
    """Parse SLAM CoNLL-U-like format into Exercise objects."""
    exercises = []
    current_meta = {}
    current_tokens = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                if current_tokens and current_meta:
                    exercises.append(Exercise(
                        user=current_meta.get("user", ""),
                        days=float(current_meta.get("days", 0)),
                        session=current_meta.get("session", ""),
                        format=current_meta.get("format", ""),
                        time=float(current_meta["time"]) if current_meta.get("time") not in (None, "", "null") else None,
                        tokens=current_tokens,
                    ))
                    if len(exercises) >= max_exercises:
                        break
                current_tokens = []
                current_meta = {}
                parts = line[2:].split()
                for part in parts:
                    if ":" in part:
                        key, val = part.split(":", 1)
                        current_meta[key] = val

            elif line.strip() == "":
                if current_tokens and current_meta:
                    exercises.append(Exercise(
                        user=current_meta.get("user", ""),
                        days=float(current_meta.get("days", 0)),
                        session=current_meta.get("session", ""),
                        format=current_meta.get("format", ""),
                        time=float(current_meta["time"]) if current_meta.get("time") not in (None, "", "null") else None,
                        tokens=current_tokens,
                    ))
                    if len(exercises) >= max_exercises:
                        break
                current_tokens = []
                current_meta = {}
            else:
                cols = line.split()
                if len(cols) >= 7:
                    current_tokens.append(Token(
                        token_id=cols[0],
                        word=cols[1],
                        pos=cols[2],
                        morph=cols[3],
                        dep_label=cols[4],
                        dep_head=int(cols[5]),
                        label=int(cols[6]),
                    ))

    if current_tokens and current_meta and len(exercises) < max_exercises:
        exercises.append(Exercise(
            user=current_meta.get("user", ""),
            days=float(current_meta.get("days", 0)),
            session=current_meta.get("session", ""),
            format=current_meta.get("format", ""),
            time=float(current_meta["time"]) if current_meta.get("time") not in (None, "", "null") else None,
            tokens=current_tokens,
        ))

    return exercises


# ─── Skill mapping ───────────────────────────────────────────────────

SKILL_CATEGORIES = [
    "grammar:present", "grammar:past", "grammar:future",
    "grammar:articles", "grammar:prepositions",
    "vocab:1", "vocab:2", "vocab:3", "vocab:4", "vocab:5",
]

VOCAB_SKILLS = ["vocab:1", "vocab:2", "vocab:3", "vocab:4", "vocab:5"]


def _tense_from_morph(morph: str) -> Optional[str]:
    if "Tense=Pres" in morph:
        return "grammar:present"
    if "Tense=Past" in morph:
        return "grammar:past"
    if "Tense=Fut" in morph:
        return "grammar:future"
    return None


def map_token_to_skill(token: Token) -> str:
    """Map a SLAM token to one of our 10 skill categories."""
    pos = token.pos.upper()

    if pos in ("VERB", "AUX"):
        tense = _tense_from_morph(token.morph)
        if tense:
            return tense
        return "grammar:present"

    if pos in ("DET",):
        return "grammar:articles"

    if pos in ("ADP",):
        return "grammar:prepositions"

    # all other POS → vocab, distributed by word hash
    bucket = hash(token.word.lower()) % 5
    return VOCAB_SKILLS[bucket]


# ─── Build interaction histories ─────────────────────────────────────

@dataclass
class Interaction:
    """One user-skill interaction: did they get it right, and when."""
    days: float
    correct: float  # 0.0 to 1.0 (fraction correct for this skill in this exercise)


def build_histories(
    exercises: List[Exercise],
) -> Dict[str, Dict[str, List[Interaction]]]:
    """
    Build per-user, per-skill interaction histories sorted by time.

    Returns: {user_id: {skill: [Interaction, ...]}}
    """
    histories: Dict[str, Dict[str, List[Interaction]]] = defaultdict(lambda: defaultdict(list))

    for ex in exercises:
        skill_results: Dict[str, List[int]] = defaultdict(list)
        for tok in ex.tokens:
            skill = map_token_to_skill(tok)
            skill_results[skill].append(1 - tok.label)  # label 0=correct→1, label 1=mistake→0

        for skill, results in skill_results.items():
            histories[ex.user][skill].append(
                Interaction(days=ex.days, correct=float(np.mean(results)))
            )

    # sort each history by time
    for user in histories:
        for skill in histories[user]:
            histories[user][skill].sort(key=lambda x: x.days)

    return dict(histories)


# ─── MLE fitting ─────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_log_likelihood(
    params: np.ndarray,
    histories: Dict[str, Dict[str, List[Interaction]]],
    init_mastery: float = 0.25,
) -> float:
    """
    Negative log-likelihood of observed data under our IRT model.

    params = [beta, alpha_0, delta]
      - beta:    sigmoid discrimination
      - alpha_0: base learning rate
      - delta:   forgetting rate per day
    """
    beta = params[0]
    alpha_0 = params[1]
    delta = params[2]

    nll = 0.0
    n_obs = 0

    for user, skills in histories.items():
        for skill, interactions in skills.items():
            mastery = init_mastery
            count = 0

            for i, interaction in enumerate(interactions):
                # apply decay based on time gap since last interaction
                if i > 0:
                    dt = max(0.0, interaction.days - interactions[i - 1].days)
                    mastery *= (1.0 - delta) ** dt

                # predict P(correct) and compare to observation
                p = sigmoid(beta * (mastery - 0.5))
                p = np.clip(p, 1e-7, 1 - 1e-7)

                c = interaction.correct
                nll -= c * np.log(p) + (1 - c) * np.log(1 - p)
                n_obs += 1

                # update mastery via prediction-error rule
                alpha_k = alpha_0 / np.sqrt(1.0 + count)
                mastery += alpha_k * (c - p)
                mastery = np.clip(mastery, 0.0, 1.0)
                count += 1

    return nll / max(n_obs, 1)


def fit_parameters(
    histories: Dict[str, Dict[str, List[Interaction]]],
) -> Dict[str, float]:
    """Fit beta, alpha_0, delta via MLE using scipy L-BFGS-B."""

    print("Fitting parameters via MLE...")
    print(f"  Users: {len(histories)}")
    total_interactions = sum(
        len(ints) for skills in histories.values() for ints in skills.values()
    )
    print(f"  Total interactions: {total_interactions}")

    # initial guess
    x0 = np.array([5.0, 0.3, 0.01])

    # bounds: beta > 0, alpha_0 in (0, 1), delta in (0, 1)
    bounds = [(0.5, 20.0), (0.01, 1.0), (0.0001, 0.5)]

    result = minimize(
        compute_log_likelihood,
        x0,
        args=(histories,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "disp": True},
    )

    beta, alpha_0, delta = result.x
    print(f"\n  Fitted parameters:")
    print(f"    beta (sigmoid steepness) = {beta:.4f}")
    print(f"    alpha_0 (learning rate)  = {alpha_0:.4f}")
    print(f"    delta (decay per day)    = {delta:.6f}")
    print(f"    Final NLL/obs            = {result.fun:.6f}")

    return {"beta": float(beta), "alpha_0": float(alpha_0), "delta_per_day": float(delta)}


# ─── Convert to env parameters ───────────────────────────────────────

def convert_to_env_params(
    fitted: Dict[str, float], steps_per_day: float = 200.0
) -> Dict[str, float]:
    """
    Convert from 'per day' timescale to 'per timestep' timescale.

    The SLAM data measures time in days. Our environment runs 200 steps
    per episode, which we treat as one tutoring session (~1 day of practice).
    So per-timestep decay = 1 - (1 - delta_per_day)^(1/steps_per_day).
    """
    delta_per_day = fitted["delta_per_day"]
    delta_per_step = 1.0 - (1.0 - delta_per_day) ** (1.0 / steps_per_day)

    env_params = {
        "beta": fitted["beta"],
        "learning_rate": fitted["alpha_0"],
        "decay_rate": float(delta_per_step),
        "note": (
            f"Fitted from Duolingo SLAM data. "
            f"delta_per_day={delta_per_day:.6f} converted to "
            f"delta_per_step={delta_per_step:.6f} assuming "
            f"{steps_per_day:.0f} steps/day."
        ),
    }
    return env_params


# ─── Compute per-skill difficulty from data ──────────────────────────

def compute_skill_difficulties(
    histories: Dict[str, Dict[str, List[Interaction]]],
) -> Dict[str, float]:
    """Estimate per-skill difficulty as 1 - average_correctness."""
    skill_correct: Dict[str, List[float]] = defaultdict(list)

    for user, skills in histories.items():
        for skill, interactions in skills.items():
            for inter in interactions:
                skill_correct[skill].append(inter.correct)

    difficulties = {}
    for skill in SKILL_CATEGORIES:
        if skill in skill_correct and skill_correct[skill]:
            difficulties[skill] = 1.0 - float(np.mean(skill_correct[skill]))
        else:
            difficulties[skill] = 0.5

    return difficulties


# ─── Fit per-format solver parameters from data ─────────────────────

@dataclass
class FormatInteraction:
    """One user-skill interaction with exercise format metadata."""
    days: float
    correct: float
    format: str
    time: Optional[float]


def build_format_histories(
    exercises: List[Exercise],
) -> Dict[str, Dict[str, List[FormatInteraction]]]:
    """Build per-user, per-skill histories that also track exercise format."""
    histories: Dict[str, Dict[str, List[FormatInteraction]]] = defaultdict(lambda: defaultdict(list))

    for ex in exercises:
        skill_results: Dict[str, List[int]] = defaultdict(list)
        for tok in ex.tokens:
            skill = map_token_to_skill(tok)
            skill_results[skill].append(1 - tok.label)

        for skill, results in skill_results.items():
            histories[ex.user][skill].append(
                FormatInteraction(
                    days=ex.days,
                    correct=float(np.mean(results)),
                    format=ex.format,
                    time=ex.time,
                )
            )

    for user in histories:
        for skill in histories[user]:
            histories[user][skill].sort(key=lambda x: x.days)

    return dict(histories)


def fit_solver_params(
    exercises: List[Exercise],
) -> Dict[str, Dict[str, float]]:
    """
    Fit per-format effectiveness and cost from SLAM exercise data.

    Effectiveness: mean correctness gain on the NEXT encounter of the same
    skill after a struggle (current correctness < 1.0). Normalized so the
    best format has effectiveness 1.0.

    Cost: median time (seconds) spent on the exercise, linearly scaled to
    [0.1, 0.5] range.

    The 3 SLAM formats map to 5 solver modules:
      reverse_tap       -> VocabDrill (quick recognition, cheapest)
      reverse_translate  -> GrammarExplanation, FreeForm (full production)
      listen             -> MixedQuiz, SpacedRepetition (listening comprehension)
    """
    histories = build_format_histories(exercises)

    format_gains: Dict[str, List[float]] = defaultdict(list)
    format_times: Dict[str, List[float]] = defaultdict(list)

    for user, skills in histories.items():
        for skill, history in skills.items():
            for i in range(len(history) - 1):
                if history[i].correct < 1.0:
                    fmt = history[i].format
                    gain = history[i + 1].correct - history[i].correct
                    format_gains[fmt].append(gain)
                    if history[i].time is not None:
                        format_times[fmt].append(history[i].time)

    raw_effectiveness = {}
    raw_cost = {}
    for fmt in format_gains:
        raw_effectiveness[fmt] = float(np.mean(format_gains[fmt]))
        raw_cost[fmt] = float(np.median(format_times[fmt])) if format_times[fmt] else 10.0

    max_eff = max(raw_effectiveness.values()) if raw_effectiveness else 1.0
    min_time = min(raw_cost.values()) if raw_cost else 1.0
    max_time = max(raw_cost.values()) if raw_cost else 1.0
    time_range = max(max_time - min_time, 1.0)

    norm_eff = {fmt: raw_effectiveness[fmt] / max_eff for fmt in raw_effectiveness}
    norm_cost = {fmt: 0.1 + 0.4 * (raw_cost[fmt] - min_time) / time_range for fmt in raw_cost}

    # Map 3 SLAM formats → 5 solvers
    solver_params = {
        "vocab_drill": {
            "effectiveness": norm_eff.get("reverse_tap", 1.0),
            "cost": norm_cost.get("reverse_tap", 0.1),
            "source_format": "reverse_tap",
        },
        "grammar_explanation": {
            "effectiveness": norm_eff.get("reverse_translate", 0.9) * 0.95,
            "cost": norm_cost.get("reverse_translate", 0.4) * 1.1,
            "source_format": "reverse_translate (scaled: explicit grammar instruction)",
        },
        "mixed_quiz": {
            "effectiveness": norm_eff.get("listen", 0.9),
            "cost": norm_cost.get("listen", 0.5) * 0.85,
            "source_format": "listen (scaled: multi-modal quiz)",
        },
        "spaced_repetition": {
            "effectiveness": norm_eff.get("listen", 0.9) * 1.05,
            "cost": norm_cost.get("reverse_tap", 0.1) * 1.5,
            "source_format": "listen+reverse_tap blend (targeted review)",
        },
        "free_form": {
            "effectiveness": norm_eff.get("reverse_translate", 0.9) * 1.05,
            "cost": norm_cost.get("reverse_translate", 0.4) * 1.2,
            "source_format": "reverse_translate (scaled: open-ended production)",
        },
    }

    # Clamp cost to [0.1, 0.5]
    for s in solver_params:
        solver_params[s]["cost"] = float(np.clip(solver_params[s]["cost"], 0.1, 0.5))
        solver_params[s]["effectiveness"] = float(np.clip(solver_params[s]["effectiveness"], 0.1, 2.0))

    print("\n  Per-format raw statistics (struggles only, correctness < 1.0):")
    for fmt in sorted(raw_effectiveness.keys()):
        print(f"    {fmt:<25} gain={raw_effectiveness[fmt]:+.4f}  median_time={raw_cost[fmt]:.1f}s  n={len(format_gains[fmt])}")

    print("\n  Solver parameters (fitted from SLAM):")
    print(f"    {'Solver':<22} {'Effectiveness':<15} {'Cost':<10} {'Source'}")
    print("    " + "-" * 75)
    for name, params in solver_params.items():
        print(f"    {name:<22} {params['effectiveness']:.3f}          {params['cost']:.3f}     {params['source_format']}")

    return solver_params


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit IRT parameters to Duolingo SLAM data")
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to SLAM .train file (auto-detects from data/ if not provided)",
    )
    parser.add_argument(
        "--max-exercises", type=int, default=200_000,
        help="Max exercises to parse (for speed)",
    )
    parser.add_argument(
        "--max-users", type=int, default=2000,
        help="Max users to use for fitting (for speed)",
    )
    parser.add_argument(
        "--output", type=str, default="data/fitted_params.json",
        help="Output path for fitted parameters",
    )
    args = parser.parse_args()

    # find data file
    data_path = args.data_path
    if data_path is None:
        data_dir = Path("data")
        candidates = list(data_dir.glob("*.train"))
        if not candidates:
            print("ERROR: No .train file found in data/")
            print("Download the SLAM dataset from:")
            print("  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO")
            print("Place the .train file in the data/ directory and re-run.")
            return
        data_path = str(candidates[0])
        print(f"Found data file: {data_path}")

    # parse
    print(f"Parsing exercises (max {args.max_exercises})...")
    exercises = parse_slam_file(data_path, max_exercises=args.max_exercises)
    print(f"  Parsed {len(exercises)} exercises")

    # build histories
    print("Building per-user, per-skill interaction histories...")
    histories = build_histories(exercises)
    print(f"  {len(histories)} unique users")

    # subsample users for speed
    if len(histories) > args.max_users:
        users = sorted(histories.keys())[:args.max_users]
        histories = {u: histories[u] for u in users}
        print(f"  Subsampled to {len(histories)} users for fitting")

    # fit
    fitted = fit_parameters(histories)

    # convert to env timescale
    env_params = convert_to_env_params(fitted)

    # compute skill difficulties
    all_histories = build_histories(exercises)
    difficulties = compute_skill_difficulties(all_histories)

    # fit per-format solver parameters
    print("\nFitting solver parameters from exercise formats...")
    solver_params = fit_solver_params(exercises)

    # save
    output = {
        "fitted_raw": fitted,
        "env_params": env_params,
        "skill_difficulties": difficulties,
        "solver_params": solver_params,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")
    print("\nTo use these in the environment, update EnvConfig:")
    print(f"  learning_rate = {env_params['learning_rate']:.4f}")
    print(f"  decay_rate    = {env_params['decay_rate']:.6f}")
    print(f"  (and set beta = {env_params['beta']:.4f} in learner.py sigmoid)")
    print(f"\nPer-skill difficulties:")
    for skill, diff in sorted(difficulties.items()):
        print(f"  {skill:<22} {diff:.3f}")


if __name__ == "__main__":
    main()
