"""Plot training/validation losses for all model runs and save to plots/ within each checkpoint dir."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODELS_DIR = "/home/dev/workspace/SuperWater/models"
RUNS = ["large_pdbs_clustered", "conf_dataset_clustered"]

TITLES = {
    "large_pdbs_clustered": "Score model — large_pdbs_clustered",
    "conf_dataset_clustered": "Confidence model — conf_dataset_clustered",
}


def plot_epoch(df_epoch, title, out_path):
    train = df_epoch[df_epoch["split"] == "train"].copy()
    val = df_epoch[df_epoch["split"] == "val"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train["epoch"], train["loss"], label="Train", lw=1.5, color="#2563eb")
    ax.plot(val["epoch"], val["loss"], label="Val", lw=1.5, color="#dc2626")

    best_val_idx = val["loss"].idxmin()
    best_val_epoch = val.loc[best_val_idx, "epoch"]
    best_val_loss = val.loc[best_val_idx, "loss"]
    ax.axvline(best_val_epoch, color="#dc2626", lw=0.8, ls="--", alpha=0.6)
    ax.scatter([best_val_epoch], [best_val_loss], color="#dc2626", zorder=5, s=40,
               label=f"Best val {best_val_loss:.4f} @ ep{int(best_val_epoch)}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title}\nEpoch-level loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_iter(df_iter, title, out_path, max_points=4000):
    # downsample if needed to avoid overplotting
    step = max(1, len(df_iter) // max_points)
    df_s = df_iter.iloc[::step].copy()

    # global iteration index for x-axis
    df_s = df_s.reset_index(drop=True)
    df_s["global_iter"] = df_s.index * step

    # smoothed version (rolling window)
    window = max(1, len(df_s) // 60)
    df_s["smooth"] = df_s["loss"].rolling(window, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df_s["global_iter"], df_s["loss"], color="#93c5fd", lw=0.6, alpha=0.7, label="Per-iter loss")
    ax.plot(df_s["global_iter"], df_s["smooth"], color="#1d4ed8", lw=1.8, label=f"Smoothed (w={window})")

    # mark epoch boundaries
    epoch_starts = df_iter.groupby("epoch").apply(lambda g: g.index[0]).reset_index(drop=False)
    epoch_starts.columns = ["epoch", "global_idx"]
    for _, row in epoch_starts.iterrows():
        ax.axvline(row["global_idx"], color="gray", lw=0.5, alpha=0.35)

    current_epoch = int(df_iter["epoch"].max())
    ax.set_xlabel(f"Iteration (epoch boundaries marked, current epoch: {current_epoch})")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title}\nIteration-level loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    for run in RUNS:
        run_dir = os.path.join(MODELS_DIR, run)
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        epoch_csv = os.path.join(run_dir, "losses_epoch.csv")
        iter_csv = os.path.join(run_dir, "losses_iter.csv")
        title = TITLES.get(run, run)

        print(f"\n=== {run} ===")

        df_epoch = pd.read_csv(epoch_csv)
        current_epoch = int(df_epoch["epoch"].max())
        train_final = df_epoch[df_epoch["split"] == "train"]["loss"].iloc[-1]
        val_final = df_epoch[df_epoch["split"] == "val"]["loss"].iloc[-1]
        best_val = df_epoch[df_epoch["split"] == "val"]["loss"].min()
        best_val_ep = df_epoch[df_epoch["split"] == "val"]["loss"].idxmin()
        best_val_ep = df_epoch.loc[best_val_ep, "epoch"]
        print(f"  Epochs completed: {current_epoch}")
        print(f"  Latest train loss: {train_final:.4f}  |  Latest val loss: {val_final:.4f}")
        print(f"  Best val loss: {best_val:.4f} at epoch {int(best_val_ep)}")

        plot_epoch(df_epoch,
                   title,
                   os.path.join(plots_dir, "loss_epoch.png"))

        df_iter = pd.read_csv(iter_csv)
        current_iter = len(df_iter)
        print(f"  Iterations logged: {current_iter}")
        plot_iter(df_iter,
                  title,
                  os.path.join(plots_dir, "loss_iter.png"))


if __name__ == "__main__":
    main()
