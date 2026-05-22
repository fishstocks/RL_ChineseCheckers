import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float_series(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            values.append(None)
        else:
            values.append(float(value))
    return values


def smooth(values, window):
    if window <= 1:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = [v for v in values[start:i + 1] if v is not None]
        smoothed.append(sum(chunk) / len(chunk) if chunk else None)
    return smoothed


def plot_metric(ax, x_key, y_key, label, rows, window):
    x_values = to_float_series(rows, x_key)
    y_values = smooth(to_float_series(rows, y_key), window)

    filtered = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if not filtered:
        return

    xs, ys = zip(*filtered)
    ax.plot(xs, ys, label=label)


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training CSV comparisons.")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "training_logs/ppo_nosub.csv",
            "training_logs/ppo_submoves.csv",
        ],
        help="CSV files to compare.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["iteration", "total_timesteps", "time_elapsed"],
        default="total_timesteps",
        help="Which x-axis to use.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Smoothing window size.",
    )
    parser.add_argument(
        "--output",
        default="training_logs/training_comparison.png",
        help="Where to save the figure.",
    )
    args = parser.parse_args()

    datasets = []
    for path in args.csv:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        label = os.path.splitext(os.path.basename(path))[0]
        datasets.append((label, load_csv(path)))

    if not datasets:
        raise FileNotFoundError("No CSV files were found to plot.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("completion_rate", "Completion Rate"),
        ("ep_len_mean", "Avg Episode Length"),
        ("ep_rew_mean", "Avg Episode Reward"),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        for label, rows in datasets:
            plot_metric(ax, args.x_axis, metric_key, label, rows, args.window)
        ax.set_title(title)
        ax.set_xlabel(args.x_axis.replace("_", " ").title())
        ax.set_ylabel(title)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    plt.show()
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
