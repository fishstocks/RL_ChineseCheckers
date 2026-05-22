# =============================================================
# player.py — simplified client that relies ONLY on JSON state
# =============================================================

import os
import json
import random
import socket
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rlchinesecheckers-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rlchinesecheckers-cache")

try:
    from sb3_contrib import MaskablePPO
except Exception as e:
    MaskablePPO = None
    PPO_IMPORT_ERROR = e
else:
    PPO_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR
TEACHER_DIR = SCRIPT_DIR / "single system"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(TEACHER_DIR) not in sys.path:
    sys.path.append(str(TEACHER_DIR))

try:
    from checkers_board import HexBoard
except Exception as e:
    HexBoard = None
    BOARD_IMPORT_ERROR = e
else:
    BOARD_IMPORT_ERROR = None

HOST = os.getenv("CHECKERS_HOST", "10.245.30.28")
PORT = int(os.getenv("CHECKERS_PORT", "50555"))
DEBUG_NET = os.getenv("DEBUG_NET", "0") not in ("0", "", "false", "False")
MODEL_PATH = Path(os.getenv("PPO_MODEL_PATH", REPO_DIR / "ppo_1mil_1v1.zip"))

DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
]
N_PINS = 10
ACTIONS_PER_PIN = len(DIRECTIONS) * 2
END_TURN_ACTION = N_PINS * ACTIONS_PER_PIN
ACTION_DIM = END_TURN_ACTION + 1
BOARD = None


def debug(*args):
    if DEBUG_NET:
        print("[NET]", *args)


def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send JSON to server and receive JSON reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}

    s.sendall(json.dumps(payload).encode("utf-8"))
    data = s.recv(1_000_000)
    s.close()

    if not data:
        return {"ok": False, "error": "no-response"}

    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


# =============================================================
# Simple renderer for the server's JSON board (optional)
# =============================================================
def render_json_board(state):
    """
    Rudimentary visualization using only JSON information.
    Does NOT require HexBoard or Pin.
    """
    pins = state.get("pins", {})
    print("=== BOARD STATE ===")
    for colour, indices in pins.items():
        print(f"{colour}: {indices}")
    print("===================")


def load_ppo_model():
    if MaskablePPO is None:
        print(f"PPO unavailable, using random legal moves: {PPO_IMPORT_ERROR}")
        return None
    try:
        model = MaskablePPO.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Could not load PPO model at {MODEL_PATH}, using random legal moves: {e}")
        return None

    obs_shape = getattr(model.observation_space, "shape", None)
    action_n = getattr(model.action_space, "n", None)
    expected_obs_shape = (len(get_board().cells) * 4,)
    if obs_shape != expected_obs_shape or action_n != ACTION_DIM:
        print(
            "PPO model shape mismatch, using random legal moves: "
            f"expected obs {expected_obs_shape}, actions {ACTION_DIM}; "
            f"got obs {obs_shape}, actions {action_n}"
        )
        return None

    print(f"Loaded PPO model: {MODEL_PATH}")
    return model


def get_board():
    global BOARD
    if BOARD is not None:
        return BOARD
    if HexBoard is None:
        raise RuntimeError(f"Could not import HexBoard: {BOARD_IMPORT_ERROR}")
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        BOARD = HexBoard(R=4, hole_radius=16, spacing=34)
    return BOARD


def normalize_legal_moves(legal_moves):
    normalized = {}
    for pid, moves in legal_moves.items():
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        normalized[pid] = [int(move) for move in moves]
    return normalized


def build_ppo_observation(state, colour):
    board = get_board()
    n_cells = len(board.cells)
    pins = state.get("pins", {})

    my_layer = np.zeros(n_cells, dtype=np.float32)
    for idx in pins.get(colour, []):
        my_layer[int(idx)] = 1.0

    enemy_layer = np.zeros(n_cells, dtype=np.float32)
    for other_colour, indices in pins.items():
        if other_colour == colour:
            continue
        for idx in indices:
            enemy_layer[int(idx)] = 1.0

    target_layer = np.zeros(n_cells, dtype=np.float32)
    target_colour = board.colour_opposites.get(colour)
    if target_colour:
        for idx in board.axial_of_colour(target_colour):
            target_layer[idx] = 1.0

    active_jump_layer = np.zeros(n_cells, dtype=np.float32)
    return np.concatenate(
        [my_layer, enemy_layer, target_layer, active_jump_layer]
    ).astype(np.float32)


def action_destination(state, colour, action):
    if action == END_TURN_ACTION:
        return None, None
    board = get_board()
    pin_id = int(action) // ACTIONS_PER_PIN
    local = int(action) % ACTIONS_PER_PIN
    direction_idx = local // 2
    is_jump = bool(local % 2)

    colour_pins = state.get("pins", {}).get(colour, [])
    if pin_id < 0 or pin_id >= len(colour_pins):
        return None, None

    src_idx = int(colour_pins[pin_id])
    src_cell = board.cells[src_idx]
    dq, dr = DIRECTIONS[direction_idx]
    multiplier = 2 if is_jump else 1
    dest_idx = board.index_of.get((src_cell.q + dq * multiplier, src_cell.r + dr * multiplier))
    if dest_idx is None:
        return None, None
    return pin_id, int(dest_idx)


def build_ppo_action_mask(state, colour, legal_moves):
    legal_moves = normalize_legal_moves(legal_moves)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for action in range(END_TURN_ACTION):
        pin_id, dest_idx = action_destination(state, colour, action)
        if pin_id is None:
            continue
        if dest_idx in legal_moves.get(pin_id, []):
            mask[action] = True
    return mask


def fallback_move(legal_moves):
    movable = [(pid, moves) for pid, moves in normalize_legal_moves(legal_moves).items() if moves]
    if not movable:
        return None, None
    pid, moves = random.choice(movable)
    return int(pid), int(random.choice(moves))


def choose_move_with_ppo(model, state, colour, legal_moves):
    if model is None:
        return fallback_move(legal_moves)

    try:
        obs = build_ppo_observation(state, colour)
        mask = build_ppo_action_mask(state, colour, legal_moves)
    except Exception as e:
        print(f"PPO adapter failed, using random legal move: {e}")
        return fallback_move(legal_moves)

    if not mask.any():
        print("PPO mask had no legal server moves, using random legal move.")
        return fallback_move(legal_moves)

    try:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    except Exception as e:
        print(f"PPO prediction failed, using random legal move: {e}")
        return fallback_move(legal_moves)

    pid, to_index = action_destination(state, colour, int(action))
    normalized = normalize_legal_moves(legal_moves)
    if pid is None or to_index not in normalized.get(pid, []):
        print(f"PPO chose non-server-legal action {int(action)}, using random legal move.")
        return fallback_move(legal_moves)

    print(f"PPO action {int(action)} -> pin {pid}, to {to_index}")
    return pid, to_index


# =============================================================
# Main client loop
# =============================================================
def main():
    timeoutnotice_move = -1
    model = load_ppo_model()
    print("==== Player ====")
    name = input("Enter name: ").strip()
    if not name:
        return

    # JOIN GAME
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    print(f"Joined game {game_id} as {colour}")

    # Wait until game ready
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    #input("Press ENTER to send START...")
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    # Wait until PLAYING
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("=== GAME STARTED ===\n")

    last_move_seen = 0

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]

        # Timeout messages
        if state.get("turn_timeout_notice") and timeoutnotice_move< state.get("move_count"):
            print("⚠ TIMEOUT:", state["turn_timeout_notice"])
            timeoutnotice_move =  state.get("move_count")


        # Finished?
        if state["status"] == "FINISHED":
            print("\n=== GAME FINISHED ===")
            print("FINAL SCORES:")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(
                        f"{pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f} "
                        f"[time={sc['time_score']:.1f}, "
                        f"moves({sc['moves']})={sc['move_score']:.1f}, "
                        f"pins={sc['pin_goal_score']:.1f}, "
                        f"dist={sc['distance_score']:.1f}]"
                    )
            print("======================")
            break

        # Render board from JSON-only
        

        # Show last move
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}→{mv['to']}  [{mv['move_ms']:.1f}ms]"
                )
            last_move_seen = state["move_count"]

        
        # If it's our turn, choose a PPO-guided move when available.
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            print("\nMy turn")
            '''------------PLAYING LOGIC-----------'''
            # Request legal moves for each pin from server
            legal_req = rpc({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id
            })

            if not legal_req.get("ok"):
                print("Error requesting legal moves:", legal_req.get("error"))
                time.sleep(0.5)
                continue

            legal_moves = legal_req.get("legal_moves", {})

            # legal_moves example structure:
            # { pin_id: [to_index1, to_index2, ...], ... }

            pid, to_index = choose_move_with_ppo(model, state, colour, legal_moves)
            if pid is None:
                print("No legal moves available.")
                time.sleep(0.5)
                continue

            delay = float(os.getenv("MOVE_DELAY_SEC", "0.2"))
            if delay > 0:
                print("Move delay:", delay)
                time.sleep(delay)
            '''-----------------PLAYING LOGIC----------------'''

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pid,
                "to_index": to_index
            })
            render_json_board(state)
            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
            else:
                if mv.get("status") == "WIN":
                    print("YOU WIN!")
                    print(mv.get("msg"))
                elif mv.get("status") == "DRAW":
                    print("DRAW")
                    print(mv.get("msg"))

        time.sleep(0.5)


if __name__ == "__main__":
    main()
