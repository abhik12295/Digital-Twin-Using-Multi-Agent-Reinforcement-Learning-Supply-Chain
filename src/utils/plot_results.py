import pandas as pd
import matplotlib.pyplot as plt


def plot_results(csv_path="results/comparison.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(df["model"], df["avg_reward"])
    plt.title("Reward Comparison")

    plt.subplot(1, 2, 2)    
    plt.bar(df["model"], df["avg_on_time_rate"])
    plt.title("On-Time Rate")

    plt.tight_layout()
    plt.savefig("results/comparison.png")
    plt.show()