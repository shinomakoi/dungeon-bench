"""Render a simple grid map from a CSV file in pygame.

Each cell's character maps to a color:
    X: White, P: Blue, M: Red, S: Purple, F: Green, W: Grey

The 'P' cell is the player and can be moved one cell at a time
horizontally and vertically with the arrow keys or WASD. Walls (W) block
movement. Reaching the green 'F' cell completes the level.

Press SPACE to let an LLM (OpenAI) drive the player automatically: the current
map is sent to the model as a markdown table and it replies with the next move to make.
"""

import csv
import json
import os
import re
import threading
from datetime import datetime

import openai
import pygame

# --- Configuration -----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# I have to send tools parameter with some models for reasoning preserve, so this is a dummy for now.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def load_settings():
    """Load settings.json (map CSV, API URL, model) next to this script.

    Missing keys fall back to defaults; a missing or invalid file is ignored."""
    defaults = {
        "map_csv": "games/easy/easy_1.csv",
        "api_url": "http://127.0.0.1:8080/v1/",
        "model": "qwen-3.8-27b",
    }
    try:
        with open(os.path.join(BASE_DIR, "settings.json"), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            defaults.update(data)
    except (OSError, ValueError):
        pass
    return defaults


SETTINGS = load_settings()

map_csv = SETTINGS["map_csv"]
CSV_PATH = map_csv if os.path.isabs(map_csv) else os.path.join(BASE_DIR, map_csv)
CELL_SIZE = 50
PADDING = 28  # margin around the grid, room for A-J / 1-10 coordinate labels
FPS = 60
PANEL_WIDTH = 260
# LLM endpoint/model come from settings.json; OPENAI_MODEL env var still wins.
LLM_API_URL = SETTINGS["api_url"]
LLM_MODEL = os.environ.get("OPENAI_MODEL", SETTINGS["model"])
LLM_RETRY_MS = 1500  # pause before re-querying after a bad/failed LLM move
MAX_ILLEGAL_MOVES = 3  # a run fails at this many illegal moves
MAX_LLM_QUERIES = 100  # a run fails once this many LLM queries have been made

BG_COLOR = (16, 18, 24)
MAP_BG = (24, 27, 35)
PANEL_BG = (30, 33, 43)
PANEL_EDGE = (52, 57, 72)
TEXT_COLOR = (225, 228, 236)
MUTED_COLOR = (128, 135, 152)
HEADER_COLOR = (90, 180, 250)
WIN_COLOR = (80, 220, 140)
PATH_COLOR = (80, 200, 210)  # Cyan tint for the LLM's planned path
GRID_LINE = (12, 14, 19)
FLOOR_DARK = (32, 36, 46)
FLOOR_LIGHT = (38, 43, 55)
WALL_HATCH = (225, 228, 236)

# Value -> color (R, G, B)
COLORS = {
    "X": FLOOR_DARK,
    "P": (70, 130, 255),  # Player blue
    "M": (235, 80, 80),  # Monster red
    "S": (240, 195, 60),  # Weapon gold
    "F": (60, 200, 120),  # Finish green
    "W": (83, 91, 112),  # Wall slate
}
DEFAULT_COLOR = (20, 22, 28)
# Direction -> (row delta, col delta) for LLM move replies.
DIRECTION_DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

SYSTEM_PROMPT = """
You are an expert Game AI Agent. Your goal is to navigate a 2D grid map to reach the Finish (F).

### Map Legend:
- X: Empty cell (passable)
- P: Your current position (Player)
- W: Wall (impassable)
- S: Weapon (must be collected to defeat Monster)
- M: Monster (blocks path; cannot be passed unless you have the Weapon)
- F: Finish (goal)

### Rules of Movement:
1. You can move one cell at a time: Up, Down, Left, or Right.
2. You cannot move into or through a Wall (W).
3. You cannot move into a cell occupied by a Monster (M) unless you have already visited the Weapon (S) cell.
4. Reaching the Finish (F) before defeating the Monster (M) (which requires the Weapon (S)) fails the game immediately.
5. The game also fails immediately if you make 3 illegal moves (moving into a wall, out of bounds, into the Monster without the Weapon, or an unparsable reply).
6. The map is a grid where columns are letters (A-J) and rows are numbers (1-10).

### Strategy Guidelines:
1. Analyze the map to locate P, S, M, and F.
2. Check if you currently possess the weapon.
3. You need a weapon to kill the monster.
4. Identify your current target.
5. You are limited to 64k tokens. Reach the Finish within the limit.

### Objective:
1. Kill the monster with weapon.
2. Head to the finish.

### Output Format:
You must output your response in JSON format only.
{"next_move_cell": "<UP|DOWN|LEFT|RIGHT>", "path": ["<direction>", "...", "..."]}
- "next_move_cell": The direction of your immediate next move (UP, DOWN, LEFT, or RIGHT).
- "path": The full planned sequence of directions from your current position to the current target (S, M, or F).
"""

USER_PROMPT_TEMPLATE = """### Current Map (markdown table):
{map_table}

### Game State:
- Current Position: {current_position}
- Inventory: {inventory}
- Target: {target}

Please analyze the map and provide the next move.
"""

LEGEND = [
    ("P", "Player (P)"),
    ("S", "Weapon (S)"),
    ("M", "Monster (M)"),
    ("W", "Wall (W)"),
    ("F", "Finish (F)"),
    ("PATH", "LLM planned path"),
]
CONTROLS = ["SPACE  LLM auto-play", "R      reset", "arrows/WASD  manual"]


def load_grid(path):
    """Load the grid from a CSV file. Returns (grid, player_row, player_col)."""
    grid = []
    player = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, 1):
            row = [cell.strip() for cell in row]
            if not any(row):  # skip whole-empty rows only
                continue
            if grid and len(row) != len(grid[0]):
                raise ValueError(
                    f"Row {idx} has {len(row)} cells; expected {len(grid[0])}"
                )
            grid.append(row)
            if player is None:
                for col, val in enumerate(row):
                    if val == "P":
                        player = (len(grid) - 1, col)
                        break
    if player is None:
        raise ValueError("No player ('P') found in grid")
    return grid, player[0], player[1]


def cell_ref_from_rc(row, col):
    """Return the LLM/human name of a cell, e.g. (2, 2) -> 'C3'."""
    return f"{chr(ord('A') + col)}{row + 1}"


def write_report(report):
    """Append one finished run's stats to reports/<model>/<map filename>.

    E.g. reports/deepseek-v4-flash/easy_1.csv, one JSON object per line."""
    model_dir = re.sub(r"[^A-Za-z0-9._-]", "_", report["model"])  # safe directory name
    map_name = os.path.basename(report["map"])
    report_dir = os.path.join(BASE_DIR, "reports", model_dir)
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, map_name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
    return path


def _reply_as_dict(raw):
    """Parse an LLM reply into its first JSON object, or None when absent."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):  # strip an optional markdown code fence
        raw = raw.strip("`").strip()
        if raw[:4].lower() == "json":
            raw = raw[4:].strip()
    data = None
    try:
        data = json.loads(raw)
    except ValueError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # grab the first JSON object
        if match:
            try:
                data = json.loads(match.group(0))
            except ValueError:
                data = None
    return data if isinstance(data, dict) else None


def parse_next_move(raw):
    """Pull the next_move_cell value out of an LLM reply as a direction name.

    Returns 'UP', 'DOWN', 'LEFT', or 'RIGHT'. Handles a bare JSON object,
    markdown-fenced JSON, or a loose direction word. Returns None when no
    usable direction is found.
    """
    data = _reply_as_dict(raw)
    if data is not None:
        val = data.get("next_move_cell")
        if isinstance(val, str):
            direction = val.strip().upper()
            if direction in DIRECTION_DELTAS:
                return direction
    raw = (raw or "").strip().upper()
    for direction in DIRECTION_DELTAS:  # last resort: a bare direction word
        if re.search(rf"\b{direction}\b", raw):
            return direction
    return None


def parse_planned_path(raw, p_row, p_col):
    """Extract the reply's 'path' list of directions as (row, col) tuples,
    resolved from the player's current position.

    Non-direction entries are skipped; returns [] when the reply has no
    usable path."""
    data = _reply_as_dict(raw)
    if data is None:
        return []
    path = data.get("path")
    if not isinstance(path, list):
        return []
    cells = []
    r, c = p_row, p_col
    for step in path:
        if not isinstance(step, str):
            continue
        delta = DIRECTION_DELTAS.get(step.strip().upper())
        if delta is None:
            continue
        r += delta[0]
        c += delta[1]
        cells.append((r, c))
    return cells


def build_prompt(snapshot, rows, cols, p_row, p_col, has_weapon):
    """Render the board as a markdown table: column letters across the top,
    row numbers down the left. Also reports live game state: the player's
    position, weapon status, and the current objective (derived from which
    clearable cells remain on the map)."""
    lines = [
        "|   | " + " | ".join(chr(ord("A") + c) for c in range(cols)) + " |",
        "|---|" + "---|" * cols,
    ]
    for r in range(rows):
        lines.append(f"| {r + 1} | " + " | ".join(snapshot[r]) + " |")
    map_table = "\n".join(lines)
    remaining = {cell for row in snapshot for cell in row}
    if "S" in remaining:
        target = "Collect Weapon (S)"
    elif "M" in remaining:
        target = "Defeat Monster (M)"
    else:
        target = "Reach Finish (F)"
    inventory = "Weapon" if has_weapon else "No Weapon"
    return USER_PROMPT_TEMPLATE.format(
        map_table=map_table,
        current_position=cell_ref_from_rc(p_row, p_col),
        inventory=inventory,
        target=target,
    )


def ask_llm(client, messages, p_row, p_col):
    """Send the accumulated conversation to the LLM and parse the reply.

    ``messages`` is the full chat history (system prompt plus prior turns).
    Returns ``(result, raw, reasoning, usage)`` where ``result`` is one of:
        ("ok", direction, path)   a valid direction plus the planned route
        ("badmove", ref)          the reply could not be turned into a valid move
        ("error", message)        the API call itself failed
    ``path`` is the model's planned route as (row, col) tuples (may be []), and
    ``raw`` is the model's raw reply text ("" when the call errored), and
    ``usage`` is a dict with prompt/completion/total tokens (None on error).
    """
    if client is None:
        return (
            ("error", "LLM client unavailable (check settings.json / API key)"),
            "",
            "",
            None,
        )
    # print(messages[-1]["content"])  # debug: the fresh map prompt
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, reasoning_effort="medium", tools=tools
        )
    except Exception as exc:  # network, auth, or API errors
        return ("error", str(exc)), "", "", None
    reply = response.choices[0].message
    raw = (reply.content or "").strip()
    reasoning = (getattr(reply, "reasoning_content", None) or "").strip()
    usage_obj = getattr(response, "usage", None)
    usage = (
        {
            "prompt": getattr(usage_obj, "prompt_tokens", 0) or 0,
            "completion": getattr(usage_obj, "completion_tokens", 0) or 0,
            "total": getattr(usage_obj, "total_tokens", 0) or 0,
        }
        if usage_obj
        else None
    )
    print(raw)  # debug: raw model reply
    direction = parse_next_move(raw)
    if direction is None:
        return ("badmove", raw or "(empty reply)"), raw, reasoning, usage
    return (
        ("ok", direction, parse_planned_path(raw, p_row, p_col)),
        raw,
        reasoning,
        usage,
    )


def main():
    pygame.init()
    grid, start_row, start_col = load_grid(CSV_PATH)
    rows = len(grid)
    cols = len(grid[0])
    p_row, p_col = start_row, start_col
    player_px = (
        PADDING + start_col * CELL_SIZE + CELL_SIZE // 2
    )  # animated token position
    player_py = PADDING + start_row * CELL_SIZE + CELL_SIZE // 2
    start_grid = [row[:] for row in grid]  # pristine copy for resets
    # Objectives present on this map; all must be cleared before the finish
    # can count (order: weapon -> monster -> finish).
    need_weapon = any("S" in row for row in start_grid)
    need_monster = any("M" in row for row in start_grid)

    map_w = cols * CELL_SIZE + PADDING * 2
    map_h = rows * CELL_SIZE + PADDING * 2
    width = map_w + PANEL_WIDTH
    height = max(map_h, 400)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DungeonBench V1")
    clock = pygame.time.Clock()

    font_small = pygame.font.SysFont(None, 22)
    font_large = pygame.font.SysFont(None, 48)
    font_cell = pygame.font.SysFont(None, 32)
    font_title = pygame.font.SysFont(None, 28)
    font_coord = pygame.font.SysFont(None, 20)
    has_weapon = False
    won = False
    failed = False
    fail_reason = ""
    moves = 0
    illegal_moves = 0
    llm_stats = {
        "queries": 0,
        "errors": 0,
        "prompt_tokens": 0,  # summed across calls (history is re-sent each turn)
        "completion_tokens": 0,  # summed across calls
        # total_tokens reported by the API for the most recent call, i.e. the
        # conversation size at that point; NOT a sum (summing per-call totals
        # would re-count the whole history on every call).
        "api_total_tokens": 0,
    }
    reported = False  # True once the end-of-run report has been written
    run_start_ms = 0  # when the current LLM run started (0 for manual play)
    run_mode = "manual"  # "manual" or "llm": how the current run is being played
    log = []

    # LLM auto-play state. The client reads OPENAI_API_KEY from the environment.
    try:
        llm_client = openai.OpenAI(base_url=LLM_API_URL, timeout=360)
    except Exception:  # no API key / bad config -> feature is disabled
        llm_client = None
    playing = False  # LLM auto-play active
    # ``messages`` holds the running conversation (system prompt + prior turns)
    # so each move builds on the model's previous answers instead of restarting.
    llm_state = {
        "busy": False,
        "result": None,
        "gen": 0,
        "next_ok": 0,
        "illegal": False,  # set when the model's last move was rejected
        "planned_path": None,  # (row, col) tuples from the last LLM reply
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    }

    def reset():
        nonlocal p_row, p_col, player_px, player_py, has_weapon, won, failed, fail_reason, playing, moves, illegal_moves
        nonlocal reported, run_start_ms, run_mode
        run_mode = "manual"
        p_row, p_col = start_row, start_col
        player_px = PADDING + start_col * CELL_SIZE + CELL_SIZE // 2
        player_py = PADDING + start_row * CELL_SIZE + CELL_SIZE // 2
        has_weapon = False
        won = False
        failed = False
        fail_reason = ""
        playing = False
        llm_state["result"] = None
        llm_state["illegal"] = False
        llm_state["planned_path"] = None
        llm_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
        llm_state["gen"] += 1  # invalidate any in-flight LLM result
        log.clear()
        moves = 0
        illegal_moves = 0
        for key in (
            "queries",
            "errors",
            "prompt_tokens",
            "completion_tokens",
            "api_total_tokens",
        ):
            llm_stats[key] = 0
        reported = False
        run_start_ms = 0
        # Restore the map to its original layout (S/M cells reappear).
        for r in range(rows):
            for c in range(cols):
                grid[r][c] = start_grid[r][c]

    def start_playback():
        """Reset and let the LLM drive the player one move at a time."""
        nonlocal playing, run_start_ms, run_mode
        reset()
        run_mode = "llm"
        run_start_ms = pygame.time.get_ticks()  # time the run for the report
        playing = True
        start_llm_query()

    def start_llm_query():
        """Snapshot the map and ask the LLM for the next move in a worker thread."""
        if not playing or won or llm_state["busy"] or llm_client is None:
            return
        # Mirror how the map is drawn: exactly one 'P' (the player). The grid
        # keeps 'P' at the origin cell, so blank any non-player 'P' to 'X'
        # before labelling the player's real position.
        snapshot = [row[:] for row in grid]
        for r in range(rows):
            for c in range(cols):
                if (r, c) != (p_row, p_col) and snapshot[r][c] == "P":
                    snapshot[r][c] = "X"
        snapshot[p_row][p_col] = "P"

        content = build_prompt(snapshot, rows, cols, p_row, p_col, has_weapon)
        if llm_state["illegal"]:  # once-per-turn notice of the rejected move
            content += "\n\nYour previous move was illegal. Pick another move."
            llm_state["illegal"] = False
        user_msg = {"role": "user", "content": content}
        gen = llm_state["gen"]
        llm_state["busy"] = True
        llm_state["next_ok"] = pygame.time.get_ticks()

        def worker():
            messages = llm_state["messages"] + [user_msg]
            result, raw, reasoning, usage = ask_llm(llm_client, messages, p_row, p_col)
            llm_stats["queries"] += 1  # every real API call counts, even stale ones
            if result[0] == "error":
                llm_stats["errors"] += 1
            elif usage:
                llm_stats["prompt_tokens"] += usage["prompt"]
                llm_stats["completion_tokens"] += usage["completion"]
                # The conversation only grows, so max() guards against a stale
                # reply arriving after a newer one.
                llm_stats["api_total_tokens"] = max(
                    llm_stats["api_total_tokens"], usage["total"]
                )
            if gen == llm_state["gen"]:  # drop results from a stale run
                # Grow the conversation with this turn so the model builds on its
                # own prior moves; skip incomplete or errored exchanges.
                if result[0] != "error" and raw:
                    llm_state["messages"].append(user_msg)
                    assistant_msg = {"role": "assistant", "content": raw}
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning
                    llm_state["messages"].append(assistant_msg)
                llm_state["result"] = result
            llm_state["busy"] = False

        threading.Thread(target=worker, daemon=True).start()

    def register_illegal_move():
        """Count an illegal move; the run fails at MAX_ILLEGAL_MOVES of them."""
        nonlocal illegal_moves, failed, fail_reason, playing
        illegal_moves += 1
        if illegal_moves >= MAX_ILLEGAL_MOVES and not failed:
            failed = True
            fail_reason = f"{MAX_ILLEGAL_MOVES} illegal moves"
            playing = False
            log.append((f"Failed: {fail_reason}", (255, 80, 80)))

    def handle_llm_result(result):
        """Apply (or log) one LLM outcome. Runs on the main thread."""
        nonlocal playing, failed
        kind = result[0]
        if kind == "ok":
            _, direction, path = result
            dr, dc = DIRECTION_DELTAS[direction]
            before = (p_row, p_col)
            try_move(dr, dc)
            if (p_row, p_col) == before:
                # Blocked by a wall/edge/monster -> pace the retry.
                log.append((f"LLM: {direction} blocked", (255, 200, 0)))
                llm_state["illegal"] = True
                llm_state["planned_path"] = None  # stale route, drop the tint
                llm_state["next_ok"] = pygame.time.get_ticks() + LLM_RETRY_MS
            else:
                llm_state["planned_path"] = path
                log.append((f"LLM: -> {direction}", TEXT_COLOR))
        elif kind == "badmove":
            log.append((f"LLM bad move: {result[1]!r}", (255, 200, 0)))
            register_illegal_move()
            llm_state["planned_path"] = None  # no valid route this turn
            llm_state["next_ok"] = pygame.time.get_ticks() + LLM_RETRY_MS
        else:  # "error"
            log.append((f"LLM error: {result[1]}", (255, 80, 80)))
            llm_state["next_ok"] = pygame.time.get_ticks() + LLM_RETRY_MS
        if won or failed:
            playing = False

    def try_move(dr, dc):
        """Move the player by (dr, dc) unless blocked by a wall, the grid
        edge, or a monster (without the weapon)."""
        nonlocal p_row, p_col, has_weapon, won, failed, playing, moves
        if won or failed:
            return
        new_row, new_col = p_row + dr, p_col + dc
        if not (0 <= new_row < rows and 0 <= new_col < cols):
            register_illegal_move()
            return
        if grid[new_row][new_col] == "W":
            register_illegal_move()
            return
        if grid[new_row][new_col] == "M" and not has_weapon:
            register_illegal_move()
            return
        moves += 1
        p_row, p_col = new_row, new_col
        cell = grid[p_row][p_col]
        if cell == "S":
            grid[p_row][p_col] = "X"
            has_weapon = True
            log.append(("Weapon acquired", (255, 215, 0)))
        elif cell == "M":
            grid[p_row][p_col] = "X"
            log.append(("Monster killed", (255, 80, 80)))
        elif cell == "F":
            # Finishing before the weapon/monster objectives are cleared is a
            # fail: the objectives must be completed in order.
            if (need_weapon and not has_weapon) or (
                need_monster and any("M" in row for row in grid)
            ):
                failed = True
                fail_reason = (
                    "objectives not completed in order (weapon -> monster -> finish)"
                )
                playing = False
                log.append((f"Failed: {fail_reason}", (255, 80, 80)))
            else:
                won = True
                log.append(("Level complete!", WIN_COLOR))

    def draw_panel():
        """Draw the header, legend, controls, live stats, and event log on the right."""
        x = map_w
        pygame.draw.rect(screen, PANEL_BG, (x, 0, PANEL_WIDTH, height))
        pygame.draw.line(screen, PANEL_EDGE, (x, 0), (x, height), 2)
        y = 18

        title = font_title.render("Dungeon Bench V1", True, HEADER_COLOR)
        screen.blit(title, (x + 16, y))
        y += 30
        model = font_small.render(f"model: {LLM_MODEL}", True, MUTED_COLOR)
        screen.blit(model, (x + 16, y))
        y += 30

        # Legend: a real color swatch next to each entry
        for key, label in LEGEND:
            swatch = pygame.Rect(x + 16, y + 4, 14, 14)
            pygame.draw.rect(screen, FLOOR_DARK, swatch)
            pygame.draw.rect(
                screen, PATH_COLOR if key == "PATH" else COLORS[key], swatch
            )
            pygame.draw.rect(screen, PANEL_EDGE, swatch, 1)
            screen.blit(font_small.render(label, True, TEXT_COLOR), (x + 40, y))
            y += 26
        y += 6
        for hint in CONTROLS:
            screen.blit(font_small.render(hint, True, MUTED_COLOR), (x + 16, y))
            y += 26
        y += 6

        # LLM status line with a pulsing dot while a query is in flight
        if playing:
            busy = llm_state["busy"]
            dot = (
                (80, 220, 140)
                if (not busy or (pygame.time.get_ticks() // 350) % 2)
                else (255, 200, 60)
            )
            pygame.draw.circle(screen, dot, (x + 22, y + 8), 5)
            status = "LLM thinking..." if busy else "LLM playing"
            color = WIN_COLOR if busy else TEXT_COLOR
        else:
            pygame.draw.circle(screen, MUTED_COLOR, (x + 22, y + 8), 5)
            status, color = "Manual mode", MUTED_COLOR
        screen.blit(font_small.render(status, True, color), (x + 34, y))
        y += 26

        stats = font_small.render(
            f"Moves: {moves}   Illegal: {illegal_moves}", True, TEXT_COLOR
        )
        screen.blit(stats, (x + 16, y))
        y += 26

        tokens = font_small.render(
            f"Tokens: {llm_stats['api_total_tokens']}   Calls: {llm_stats['queries']}",
            True,
            TEXT_COLOR,
        )
        screen.blit(tokens, (x + 16, y))
        y += 26

        # --- Event log ---
        pygame.draw.line(
            screen, PANEL_EDGE, (x + 16, y + 6), (x + PANEL_WIDTH - 16, y + 6)
        )
        y += 16
        header = font_small.render("EVENTS", True, HEADER_COLOR)
        screen.blit(header, (x + 16, y))
        y += 28
        if not log:
            empty = font_small.render("(none yet)", True, MUTED_COLOR)
            screen.blit(empty, (x + 16, y))
        else:
            # Show only as many recent events as fit in the panel; older
            # entries are dimmed and every entry gets a sequence number.
            max_lines = max(1, (height - y - PADDING) // 24)
            shown = log[-max_lines:]
            first = len(log) - len(shown)
            for i, (text, color) in enumerate(shown):
                fade = 1.0 if len(shown) == 1 else 0.4 + 0.6 * (i / (len(shown) - 1))
                dim = tuple(int(ch * fade) for ch in color)
                num = font_small.render(f"{first + i + 1:3d}", True, MUTED_COLOR)
                surf = font_small.render(text, True, dim)
                max_w = PANEL_WIDTH - 66
                while surf.get_width() > max_w and len(text) > 1:  # clip overflow
                    text = text[:-1]
                    surf = font_small.render(text, True, dim)
                screen.blit(num, (x + 16, y))
                screen.blit(surf, (x + 46, y))
                y += 24

    def draw_grid():
        # Map area background (slightly lighter than the window background)
        pygame.draw.rect(screen, MAP_BG, (0, 0, map_w, map_h))

        # Coordinate labels: column letters across the top, row numbers down the
        # left, matching the A-J / 1-10 cell names the LLM uses in its replies.
        for c in range(cols):
            label = font_coord.render(chr(ord("A") + c), True, MUTED_COLOR)
            screen.blit(
                label,
                label.get_rect(
                    center=(PADDING + c * CELL_SIZE + CELL_SIZE // 2, PADDING // 2)
                ),
            )
        for r in range(rows):
            label = font_coord.render(str(r + 1), True, MUTED_COLOR)
            screen.blit(
                label,
                label.get_rect(
                    center=(PADDING // 2, PADDING + r * CELL_SIZE + CELL_SIZE // 2)
                ),
            )

        path = list(llm_state["planned_path"] or [])
        if (p_row, p_col) in path:  # drop cells already walked (incl. the old cell)
            path = path[path.index((p_row, p_col)) + 1 :]
        path_set = set(path)
        path_tint = None
        if path_set:  # one reusable translucent overlay for all path cells
            path_tint = pygame.Surface((CELL_SIZE - 4, CELL_SIZE - 4), pygame.SRCALPHA)
            path_tint.fill((*PATH_COLOR, 90))
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(
                    PADDING + c * CELL_SIZE,
                    PADDING + r * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                base = grid[r][c]
                if base == "P" and (r, c) != (p_row, p_col):  # vacated origin tile
                    base = "X"
                if base in ("X", "P"):  # floor: subtle checkerboard
                    color = FLOOR_LIGHT if (r + c) % 2 else FLOOR_DARK
                else:
                    color = COLORS.get(base, DEFAULT_COLOR)
                pygame.draw.rect(screen, color, rect)
                if base == "W":  # hatched walls, no repeated letters
                    pygame.draw.line(
                        screen,
                        WALL_HATCH,
                        (rect.x + 8, rect.bottom - 8),
                        (rect.right - 8, rect.y + 8),
                    )
                    pygame.draw.line(
                        screen,
                        WALL_HATCH,
                        (rect.x + 8, rect.bottom - 18),
                        (rect.right - 18, rect.y + 8),
                    )
                if (r, c) in path_set and (r, c) != (p_row, p_col):
                    screen.blit(path_tint, (rect.x + 2, rect.y + 2))
                pygame.draw.rect(screen, GRID_LINE, rect, 1)
                if base in ("S", "M", "F"):  # letters only on the special cells
                    tcolor = (
                        (20, 22, 28) if base == "S" else (255, 255, 255)
                    )  # dark on gold
                    tsurf = font_cell.render(base, True, tcolor)
                    screen.blit(tsurf, tsurf.get_rect(center=rect.center))

        # Player token: a ringed circle at the animated pixel position
        px, py = int(player_px), int(player_py)
        radius = CELL_SIZE // 2 - 8
        pygame.draw.circle(screen, (255, 255, 255), (px, py), radius)
        pygame.draw.circle(screen, COLORS["P"], (px, py), radius - 3)
        pygame.draw.circle(
            screen, (170, 205, 255), (px - radius // 3, py - radius // 3), 3
        )

    def draw_win_overlay():
        """Show a centered 'Level complete!' notification over the map."""
        overlay = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        msg = font_large.render("Level complete!", True, WIN_COLOR)
        stats = font_small.render(
            f"Moves: {moves}    Illegal moves: {illegal_moves}", True, TEXT_COLOR
        )
        tokens = font_small.render(
            f"Tokens: {llm_stats['api_total_tokens']}    LLM calls: {llm_stats['queries']}",
            True,
            TEXT_COLOR,
        )
        sub = font_small.render("Press R to play again", True, TEXT_COLOR)
        screen.blit(msg, msg.get_rect(center=(map_w // 2, map_h // 2 - 50)))
        screen.blit(stats, stats.get_rect(center=(map_w // 2, map_h // 2 - 12)))
        screen.blit(tokens, tokens.get_rect(center=(map_w // 2, map_h // 2 + 16)))
        screen.blit(sub, sub.get_rect(center=(map_w // 2, map_h // 2 + 44)))

    def draw_fail_overlay():
        """Show a centered 'Level failed!' notification over the map."""
        overlay = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        msg = font_large.render("Level failed!", True, (255, 80, 80))
        stats = font_small.render(
            f"Moves: {moves}    Illegal moves: {illegal_moves}", True, TEXT_COLOR
        )
        sub1 = font_small.render(fail_reason or "Run failed.", True, TEXT_COLOR)
        tokens = font_small.render(
            f"Moves: {moves}  Illegal: {illegal_moves}  Tokens: {llm_stats['api_total_tokens']}",
            True,
            TEXT_COLOR,
        )
        sub2 = font_small.render("Press R to play again", True, TEXT_COLOR)
        screen.blit(msg, msg.get_rect(center=(map_w // 2, map_h // 2 - 56)))
        screen.blit(stats, stats.get_rect(center=(map_w // 2, map_h // 2 - 22)))
        screen.blit(sub1, sub1.get_rect(center=(map_w // 2, map_h // 2 + 6)))
        screen.blit(tokens, tokens.get_rect(center=(map_w // 2, map_h // 2 + 30)))
        screen.blit(sub2, sub2.get_rect(center=(map_w // 2, map_h // 2 + 54)))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    start_playback()
                elif event.key == pygame.K_r:
                    reset()
                elif not playing:  # manual keys are ignored while the LLM drives
                    if event.key in (pygame.K_UP, pygame.K_w):
                        try_move(-1, 0)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        try_move(1, 0)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        try_move(0, -1)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        try_move(0, 1)

        # --- LLM auto-play tick ---
        if playing:
            # Apply a completed LLM result, then request the next move once idle.
            if not llm_state["busy"] and llm_state["result"] is not None:
                handle_llm_result(llm_state["result"])
                llm_state["result"] = None
            if (
                playing
                and not won
                and not failed
                and not llm_state["busy"]
                and pygame.time.get_ticks() >= llm_state["next_ok"]
            ):
                if llm_stats["queries"] >= MAX_LLM_QUERIES:
                    failed = True
                    fail_reason = (
                        f"LLM query budget exhausted ({MAX_LLM_QUERIES} queries)"
                    )
                    playing = False
                    log.append((f"Failed: {fail_reason}", (255, 80, 80)))
                else:
                    start_llm_query()

        # --- End-of-run report (written once when the level is decided) ---
        if (won or failed) and not reported:
            reported = True
            report = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": LLM_MODEL,
                "mode": run_mode,
                "api_url": LLM_API_URL,
                "map": os.path.relpath(CSV_PATH, BASE_DIR),
                "result": "win" if won else "fail",
                "fail_reason": fail_reason if failed else None,
                "moves": moves,
                "illegal_moves": illegal_moves,
                "llm_queries": llm_stats["queries"],
                "llm_query_budget": MAX_LLM_QUERIES,
                "llm_errors": llm_stats["errors"],
                # "total" is the API-reported total_tokens of the final call
                # (conversation size at the end). prompt/completion are sums
                # across all calls, so they exceed "total": each call re-sends
                # the whole history as prompt.
                "tokens": {
                    "total": llm_stats["api_total_tokens"],
                    "prompt": llm_stats["prompt_tokens"],
                    "completion": llm_stats["completion_tokens"],
                },
                "duration_s": (
                    round((pygame.time.get_ticks() - run_start_ms) / 1000, 1)
                    if run_start_ms
                    else None
                ),
            }
            path = write_report(report)
            log.append((f"Report: {os.path.relpath(path, BASE_DIR)}", (0, 200, 200)))

        # --- Draw ---
        # Slide the player token toward its current cell (snap on a reset).
        target_px = PADDING + p_col * CELL_SIZE + CELL_SIZE // 2
        target_py = PADDING + p_row * CELL_SIZE + CELL_SIZE // 2
        if (
            abs(target_px - player_px) > CELL_SIZE
            or abs(target_py - player_py) > CELL_SIZE
        ):
            player_px, player_py = float(target_px), float(target_py)
        else:
            step = min(1.0, clock.get_time() / 90.0)
            player_px += (target_px - player_px) * step
            player_py += (target_py - player_py) * step

        screen.fill(BG_COLOR)
        draw_grid()  # map background is drawn by draw_grid itself
        draw_panel()
        if won:
            draw_win_overlay()
        elif failed:
            draw_fail_overlay()

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
