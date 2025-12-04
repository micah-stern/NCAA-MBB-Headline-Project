# src/preprocess.py
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


def format_game_context(game_df, headline_row):
    """Convert boxscore data to natural language prompt."""
    # Get game info from headline data
    game_id = headline_row["game_id"]
    headline = headline_row["headline"]

    # Filter boxscores for this game
    game_players = game_df[game_df["game_id"] == game_id]

    if game_players.empty:
        return None

    # Get team names from the player data
    teams = game_players["team"].unique()
    if len(teams) < 2:
        return None

    team1, team2 = teams[:2]

    # Get top performers
    try:
        top_performers = game_players.nlargest(3, "PTS")
        top_performers_str = "; ".join(
            [
                f"{row['player']} ({row['team']}): {row['PTS']} pts, {row['REB']} reb, {row['AST']} ast"
                for _, row in top_performers.iterrows()
            ]
        )
    except:
        top_performers_str = "No stats available"

    # Format input text
    input_text = (
        f"Generate a sports headline for the game between {team1} and {team2}. "
        f"Key performers: {top_performers_str}."
    )

    return {"input_text": input_text, "target_text": headline, "game_id": game_id}


def preprocess_data(boxscore_path, headline_path, output_dir="data/processed"):
    """Process and save training data."""
    # Create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    boxscores = pd.read_csv(boxscore_path)
    headlines = pd.read_csv(headline_path)

    # Filter out games without headlines
    valid_games = headlines[headlines["headline"] != "No headline found"]

    # Process each game
    training_data = []
    for _, row in tqdm(
        valid_games.iterrows(), total=len(valid_games), desc="Processing games"
    ):
        formatted = format_game_context(boxscores, row)
        if formatted:
            training_data.append(formatted)

    # Save as JSONL
    output_file = Path(output_dir) / "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nProcessed {len(training_data)} examples. Saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess NCAA basketball data")
    parser.add_argument(
        "--boxscores",
        default="data/raw/cbb_boxscores_11_2024.csv",
        help="Path to boxscores CSV",
    )
    parser.add_argument(
        "--headlines",
        default="data/raw/cbb_headlines_11_2024.csv",
        help="Path to headlines CSV",
    )
    parser.add_argument(
        "--output_dir", default="data/processed", help="Output directory"
    )
    args = parser.parse_args()

    preprocess_data(args.boxscores, args.headlines, args.output_dir)
