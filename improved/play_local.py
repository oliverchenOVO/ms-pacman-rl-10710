"""Play one local evaluation episode with the retained high-score model.

No score is submitted unless PACMAN_SCORE_SERVER_URL is explicitly set.
"""

import os

from stable_baselines3 import PPO

from game_environment import GameEnvironment


MODEL_PATH = "models/MsPacman-v5_avg10500.zip"


def main():
    player_name = os.getenv("PACMAN_PLAYER_NAME", "local-player")
    student_id = int(os.getenv("PACMAN_STUDENT_ID", "0"))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    env = GameEnvironment(
        env_id="ALE/MsPacman-v5",
        use_custom_reward=False,
        use_custom_obs=False,
    )
    env.load_model(PPO, MODEL_PATH)
    score = env.play(player_name, student_id, show_game=True)
    print(f"Final score: {score:.0f}")


if __name__ == "__main__":
    main()
