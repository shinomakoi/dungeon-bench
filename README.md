# DungeonBench V1

A grid dungeon game. The game uses pygame to show the grid on the screen. An LLM (Large Language Model) moves the player from the start to the finish. You see the player move on the screen. This tool tests how well different models make one decision at a time in a small game. 

<img width="946" height="719" alt="screenshot" src="https://github.com/user-attachments/assets/0555b538-f821-4ff9-9e63-ac498aefe1a5" />


### Leaderboard

| Model  | Score | 
|---|---|
| DeepSeek-V4-Pro (high) | 🥇12/12 |
| Gemma-4-31B-it | 🥈11/12 |
| Qwen-3.8-27B (medium) | 🥈11/12 | 
| GLM-5.3-Flash (high) | 🥈11/12 |
| Muse-Glimmer-30B (medium) | 🥉10/12 | 
| DeepSeek-V4-Flash (high) | 🥉10/12 |
| Granite 4.2 (full) | 8/12 |
| KAT-Coder-V2.5-Dev | 8/12 |
| Nemotron-3.5-Lightning-30B-A3B | 5/12 | 

| Model  | Illegal moves | 
|---|---|
| DeepSeek-V4-Pro (high) | 🥇0 |
| Gemma-4-31B-it | 🥈1 |
| Qwen-3.8-27B (medium) | 🥈1 | 
| Muse-Glimmer-30B (medium) | 🥉2| 
| Granite 4.2 (full) | 🥉2 |
| Nemotron-3.5-Lightning-30B-A3B | 7 | 
| GLM-5.3-Flash (high) | 8 |
| KAT-Coder-V2.5-Dev | 10 |
| DeepSeek-V4-Flash (high) | 12 |

## How the Game Works

The map is a grid. The game loads the grid from a CSV file in the `games/` folder.

The map uses these symbols:

- `P` - Player
- `S` - Weapon
- `M` - Monster
- `W` - Wall
- `F` - Finish

**Manual mode:** You move the player with the arrow keys or the WASD keys.

**LLM mode:** Press SPACE. The game sends the current map (as a markdown table) to the model. The model replies with the next move. The LLM plays one turn at a time. The level ends when the player wins or fails.

### Win Rules

- The player must get the Weapon (S) first.
- Next, the player must defeat the Monster (M).
- Last, the player must reach the Finish (F).
- If the player reaches the Finish (F) before the Weapon (S) or the Monster (M), the run fails.

### Fail Rules

- Three illegal moves fail the run. An illegal move is: walking into a wall, moving outside the grid, moving onto a Monster (M) without a Weapon (S), or a reply from the LLM that the game cannot read.
- The run fails if the LLM uses all 100 of its queries.

### Stats Report

When a run ends, the game adds a stats report to `reports/<model>/<map>.csv`. The report has: the number of moves, the number of illegal moves, the number of LLM calls, the number of tokens used, and the time. Each line in the file is one JSON object.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

The game loads settings from `settings.json`. Rename the example file:

```bash
cp settings.json.example settings.json
```

| Key         | Description                              | Default                     |
|-------------|------------------------------------------|-----------------------------|
| `map_csv`   | Map file to load                         | `games/easy/easy_1.csv`     |
| `api_url`   | URL where the game sends API requests    | `http://127.0.0.1:8935/v1/` |
| `model`     | Name of the model to query               | `qwen-3.8-27b`               |

The game reads the API key from the `OPENAI_API_KEY` environment variable. The `OPENAI_MODEL` environment variable overrides the `model` setting.

## Run the Game

```bash
export OPENAI_API_KEY=sk-...
python game.py
```

### Controls

- **SPACE** : Start LLM auto-play
- **R** : Reset the level
- **Arrow keys / WASD** : Move the player by hand (the game ignores these keys while the LLM plays)
- **ESC** : Quit