import os
import pandas as pd
from pathlib import Path
from src.data.live_data import LiveSupplyChainDataPipeline
from src.models.training import evaluate_random_policy, evaluate_heuristic_policy
from src.models.ppo_training import train_ppo_model, evaluate_ppo_model
from src.utils.plot_results import plot_results
from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise EnvironmentError("FRED_API_KEY is missing.")

    pipeline = LiveSupplyChainDataPipeline(fred_api_key=fred_api_key)
    feature_store = pipeline.build_feature_store()
    pipeline.save_feature_store(feature_store)

    random_metrics = evaluate_random_policy(episodes=10, seed=42)
    heuristic_metrics = evaluate_heuristic_policy(episodes=10, seed=42)

    print("=" * 72)
    print("TRAINING PPO AGENT")
    print("=" * 72)
    model = train_ppo_model(total_timesteps=10000, seed=42)

    ppo_metrics = evaluate_ppo_model(model=model, episodes=10, seed=42)

    print("\n" + "=" * 72)
    print("SPRINT 2B - POLICY COMPARISON")
    print("=" * 72)

    print("\nRandom Policy")
    print(f"Average reward          : {random_metrics['avg_reward']:.4f}")
    print(f"Average normalized cost : {random_metrics['avg_cost']:.4f}")
    print(f"Average on-time rate    : {random_metrics['avg_on_time_rate'] * 100:.2f}%")
    print(f"Average disruption score: {random_metrics['avg_disruption_score']:.4f}")

    print("\nHeuristic Policy")
    print(f"Average reward          : {heuristic_metrics['avg_reward']:.4f}")
    print(f"Average normalized cost : {heuristic_metrics['avg_cost']:.4f}")
    print(f"Average on-time rate    : {heuristic_metrics['avg_on_time_rate'] * 100:.2f}%")
    print(f"Average disruption score: {heuristic_metrics['avg_disruption_score']:.4f}")

    print("\nPPO Policy")
    print(f"Average reward          : {ppo_metrics['avg_reward']:.4f}")
    print(f"Average normalized cost : {ppo_metrics['avg_cost']:.4f}")
    print(f"Average on-time rate    : {ppo_metrics['avg_on_time_rate'] * 100:.2f}%")
    print(f"Average disruption score: {ppo_metrics['avg_disruption_score']:.4f}")

    results_df = pd.DataFrame([
        {"model": "Random", **random_metrics},
        {"model": "Heuristic", **heuristic_metrics},
        {"model": "PPO", **ppo_metrics},
    ])
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / "comparison.csv", index=False)
    plot_results()


if __name__ == "__main__":
    main()