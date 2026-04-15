import os

from src.data.live_data import LiveSupplyChainDataPipeline
from src.models.training import evaluate_random_policy
from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise EnvironmentError(
            "FRED_API_KEY is missing. Set it in your shell before running main.py"
        )

    pipeline = LiveSupplyChainDataPipeline(fred_api_key=fred_api_key)
    feature_store = pipeline.build_feature_store()
    pipeline.save_feature_store(feature_store)

    metrics = evaluate_random_policy(episodes=10, seed=42)

    print("=" * 72)
    print("SPRINT 1 COMPLETE - LIVE PUBLIC DATA INGESTION + RL ENVIRONMENT")
    print("=" * 72)
    print("Saved:")
    print(" - data/processed/fred_features.csv")
    print(" - data/processed/weather_features.csv")
    print()
    print("Baseline random policy metrics")
    print(f"Average reward          : {metrics['avg_reward']:.4f}")
    print(f"Average normalized cost : {metrics['avg_cost']:.4f}")
    print(f"Average on-time rate    : {metrics['avg_on_time_rate'] * 100:.2f}%")
    print(f"Average disruption score: {metrics['avg_disruption_score']:.4f}")

    '''
    ========================================================================
    SPRINT 1 COMPLETE - LIVE PUBLIC DATA INGESTION + RL ENVIRONMENT
    ========================================================================
    Saved:
    - data/processed/fred_features.csv
    - data/processed/weather_features.csv

    Baseline random policy metrics
    Average reward          : -16.7786
    Average normalized cost : 0.6139
    Average on-time rate    : 61.80%
    Average disruption score: 0.5346
    '''

if __name__ == "__main__":
    main()