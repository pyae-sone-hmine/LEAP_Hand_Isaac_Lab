from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(results: Dict[str, Dict], output_dir: Path, dpi: int = 300) -> None:
    """Generate comparison plots for a mapping of policy name -> metrics dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = list(results.keys())

    success_rates = [results[p]["success_rate"] for p in policies]
    times = [results[p]["time_to_goal"] for p in policies]
    errors = [results[p]["steady_state_error"] for p in policies]
    goals = [results[p]["goals_until_failure"] for p in policies]
    jitter = [results[p].get("jitter_rms_per_env", []) for p in policies]

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=dpi)
        plt.close(fig)

    def _shrink_task_labels(ax: Axes) -> None:
        """Use a smaller font for task names on the x-axis."""
        ax.tick_params(axis="x", labelsize=8)

    # Success rate
    fig, ax = plt.subplots()
    ax.bar(policies, success_rates, color="skyblue")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate per policy")
    _shrink_task_labels(ax)
    _save(fig, "success_rate.png")

    # Time to reach goal
    fig, ax = plt.subplots()
    means = [np.mean(t) if len(t) > 0 else 0.0 for t in times]
    stds = [np.std(t) if len(t) > 0 else 0.0 for t in times]
    lower_err = [min(m, s) for m, s in zip(means, stds)]  # clamp at zero
    upper_err = stds
    ax.bar(policies, means, yerr=[lower_err, upper_err], capsize=4, color="lightgreen")
    ax.set_ylabel("Time to goal (s)")
    ax.set_title("Time to reach goal")
    _shrink_task_labels(ax)
    _save(fig, "time_to_goal.png")

    # Steady-state error
    fig, ax = plt.subplots()
    err_means = [np.mean(e) if len(e) > 0 else 0.0 for e in errors]
    err_stds = [np.std(e) if len(e) > 0 else 0.0 for e in errors]
    err_lower = [min(m, s) for m, s in zip(err_means, err_stds)]  # clamp at zero
    err_upper = err_stds
    ax.bar(policies, err_means, yerr=[err_lower, err_upper], capsize=4, color="salmon")
    ax.set_ylabel("Orientation error (rad)")
    ax.set_title("Steady-state error")
    _shrink_task_labels(ax)
    _save(fig, "steady_state_error.png")

    # Goals until failure
    fig, ax = plt.subplots()
    goal_means = [np.mean(g) if len(g) > 0 else 0.0 for g in goals]
    goal_stds = [np.std(g) if len(g) > 0 else 0.0 for g in goals]
    goal_lower = [min(m, s) for m, s in zip(goal_means, goal_stds)]  # clamp at zero
    goal_upper = goal_stds
    ax.bar(policies, goal_means, yerr=[goal_lower, goal_upper], capsize=4, color="plum")
    ax.set_ylabel("Goals completed before failure")
    ax.set_title("Goals until failure")
    _shrink_task_labels(ax)
    _save(fig, "goals_until_failure.png")

    # Jitter (high-pass residual RMS)
    fig, ax = plt.subplots()
    jitter_means = [np.mean(j) if len(j) > 0 else 0.0 for j in jitter]
    jitter_stds = [np.std(j) if len(j) > 0 else 0.0 for j in jitter]
    jitter_lower = [min(m, s) for m, s in zip(jitter_means, jitter_stds)]
    jitter_upper = jitter_stds
    ax.bar(policies, jitter_means, yerr=[jitter_lower, jitter_upper], capsize=4, color="gold")
    ax.set_ylabel("Jitter RMS (rad)")
    ax.set_title("Joint jitter (residual RMS)")
    _shrink_task_labels(ax)
    _save(fig, "jitter_rms.png")


def _load_results(results_path: Path) -> Dict[str, Dict]:
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Results file is not a mapping: {results_path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from a results.json file.")
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/1d_bi_comparison/results.json",
        help="Path to results.json produced by 1D_experiments.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write plots (defaults to results file directory).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Dots per inch for saved plots (higher = higher resolution).",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    try:
        results = _load_results(results_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir is not None else results_path.parent
    plot_metrics(results, output_dir, dpi=args.dpi)
    print(f"[INFO] Saved plots to {output_dir}")


if __name__ == "__main__":
    main()

