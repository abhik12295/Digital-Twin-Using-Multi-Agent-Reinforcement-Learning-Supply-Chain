import os
from dotenv import load_dotenv

load_dotenv()

from src.data.live_data import LiveSupplyChainDataPipeline
from src.models.training import evaluate_random_policy, evaluate_heuristic_policy


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
    print("SPRINT 2A - BASELINE COMPARISON")
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
    '''
    ========================================================================
    SPRINT 2A - BASELINE COMPARISON
    ========================================================================

    Random Policy
    Average reward          : -16.7786
    Average normalized cost : 0.6139
    Average on-time rate    : 61.80%
    Average disruption score: 0.5346

    Heuristic Policy
    Average reward          : -14.3862
    Average normalized cost : 0.6320
    Average on-time rate    : 66.00%
    Average disruption score: 0.5346
    '''


if __name__ == "__main__":
    main()