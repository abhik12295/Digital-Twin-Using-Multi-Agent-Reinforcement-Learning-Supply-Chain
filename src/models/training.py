from __future__ import annotations
from typing import Dict, List
import numpy as np

from src.agents.random_agent import RandomRoutingAgent
from src.agents.heuristic_agent import HeuristicRoutingAgent
from src.simulation.environment import SupplyChainRoutingEnv


def _run_policy(env: SupplyChainRoutingEnv, agent, episodes: int, seed: int) -> Dict[str, float]:
    episode_rewards: List[float] = []
    episode_costs: List[float] = []
    episode_service: List[float] = []
    episode_disruption: List[float] = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        total_cost = 0.0
        total_on_time = 0
        total_disruption = 0.0
        steps = 0

        while not done:
            if hasattr(agent, "act") and agent.__class__.__name__ == "RandomRoutingAgent":
                action = agent.act()
            else:
                action = agent.act(state)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_cost += info["cost"]
            total_on_time += info["on_time"]
            total_disruption += info["disruption_score"]
            steps += 1

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost / max(steps, 1))
        episode_service.append(total_on_time / max(steps, 1))
        episode_disruption.append(total_disruption / max(steps, 1))

    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_cost": float(np.mean(episode_costs)),
        "avg_on_time_rate": float(np.mean(episode_service)),
        "avg_disruption_score": float(np.mean(episode_disruption)),
    }


def evaluate_random_policy(episodes: int = 10, seed: int = 42) -> Dict[str, float]:
    env = SupplyChainRoutingEnv(seed=seed)
    agent = RandomRoutingAgent(action_size=env.action_space.n, seed=seed)
    return _run_policy(env, agent, episodes, seed)


def evaluate_heuristic_policy(episodes: int = 10, seed: int = 42) -> Dict[str, float]:
    env = SupplyChainRoutingEnv(seed=seed)
    agent = HeuristicRoutingAgent()
    return _run_policy(env, agent, episodes, seed)