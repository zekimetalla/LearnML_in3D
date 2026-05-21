"""Step 1 — Collect a deliberate dataset.

Run:  python 01_collect.py --tag v1 --seed 42

It opens an interactive session, walks you through five driving phases on the
terminal, and ALSO polls (x, z) positions during your drive so you can later
overlay your training-drive path against the NN test-drive path
(see drive2win.viz.plot_path_overlay).

Output:  data_<tag>.npz with arrays `states`, `actions`, `positions`, and
the integer `seed` used for the map.
"""
from __future__ import annotations
import argparse
import threading
import time
import numpy as np

from game_client import GameClient

SERVER_URL = "https://ml.ferit.tech"
API_KEY = "None"  # paste yours if the server requires it

PHASES = [
    ("Smooth laps",       90, "Hold throttle on straights, smooth steering through corners."),
    ("Tight turns",       60, "Slow before each corner, take it cleanly."),
    ("Obstacle clusters", 60, "Brake when the front ray gets short, steer around."),
    ("Bad terrain",       60, "Drive deliberate lines on ice / mud / sand."),
    ("Recovery",          60, "Drive into walls, get stuck, back out, turn around. DO NOT SKIP."),
]


def _poll_positions(client: GameClient, stop_evt: threading.Event,
                    out: list, hz: float = 5.0):
    """Background thread: poll position at low Hz so we can plot the path later."""
    interval = 1.0 / hz
    while not stop_evt.is_set():
        try:
            st = client.get_latest_state()
            pos = st.get("position") if st else None
            if pos and "x" in pos and "z" in pos:
                out.append((time.time(), pos["x"], pos["z"]))
        except Exception:
            pass
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1",
                    help="Suffix for output file (data_<tag>.npz)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Map seed. Same seed across iterations keeps the "
                         "comparison clean; vary it once you want to test "
                         "generalisation across different terrains.")
    ap.add_argument("--no-obstacles", action="store_true",
                    help="Disable arena obstacles for clean navigation data.")
    args = ap.parse_args()

    client = GameClient(SERVER_URL, API_KEY)
    session = client.create_session(
        mode="time_trial",
        player_name=f"d2w_collector_{args.tag}",
        config={"seed": args.seed, "wind_enabled": False,
                "obstacles_enabled": not args.no_obstacles},
    )
    print("Open this URL in a NEW TAB and click into it so WASD reach the game:")
    print(" ", session.get("browser_url"))
    print()
    input("Press Enter once the browser tab has focus and you can see the bot. ")

    client.connect_ws()
    time.sleep(0.5)

    positions: list = []
    stop_evt = threading.Event()
    t = threading.Thread(target=_poll_positions, args=(client, stop_evt, positions),
                         daemon=True)
    t.start()

    client.start_recording(sample_rate=20)
    for i, (name, seconds, hint) in enumerate(PHASES, 1):
        print(f"\n--- Phase {i}/{len(PHASES)} — {name} ({seconds}s) ---")
        print(f"  {hint}")
        print(f"  Driving for {seconds}s; switch to the browser tab now.")
        for s in range(seconds, 0, -10):
            print(f"  ... {s}s remaining")
            time.sleep(min(10, s))

    stop_evt.set()
    info = client.stop_recording()
    print(f"\nStopped. Samples on the server: {info.get('sample_count', '?')}")

    states_raw, actions = client.get_recording_as_arrays()
    print(f"states shape   : {states_raw.shape}   (N, 12)")
    print(f"actions shape  : {actions.shape}      (N, 2)")

    pos_arr = np.array([(p[1], p[2]) for p in positions], dtype=np.float32)
    print(f"positions shape: {pos_arr.shape}     (M, 2)  — low-Hz path samples")

    assert states_raw.shape[0] >= 5_000, (
        "Fewer than 5,000 samples. Drive more before saving."
    )

    out = f"data_{args.tag}.npz"
    np.savez(out, states=states_raw, actions=actions, positions=pos_arr,
             seed=args.seed)
    print(f"Saved {out}")

    try:
        client.disconnect_ws()
        client.delete_session()
    except Exception:
        pass


if __name__ == "__main__":
    main()
