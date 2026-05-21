"""
ML Simulation Game Client — Python SDK

Provides a simple interface for students to interact with the 3D simulation
without worrying about HTTP requests or WebSocket management.

Usage:
    from game_client import GameClient

    client = GameClient("https://ml.ferit.tech", api_key="mlsim_abc123...")
    session = client.create_session(mode="target_practice")
    data = client.fire_projectile(angle=45, force=70)
    sensors = client.get_sensors()
    client.send_control(throttle=0.8, steering=-0.3)
"""

import json
import time
import threading
import numpy as np
import requests
import websocket


class GameClient:
    """Client for the ML simulation game server."""

    def __init__(self, server_url: str = "https://ml.ferit.tech", api_key: str = "None"):
        """
        Initialize the game client.

        Args:
            server_url: URL of the relay server (default: http://localhost:3000)
            api_key: API key for authentication (required in production)
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.session_id = None
        self._ws = None
        self._ws_thread = None
        self._latest_state = None
        self._state_lock = threading.Lock()
        self._callbacks = {}
        # Cached static world-map snapshot (terrain IDs, elevations, obstacles,
        # checkpoints). Populated by cache_world_map(); enables get_grid_local()
        # to compute the heading-aligned 32×32 grid offline at 20 Hz.
        self._world_map = None

        # Use a requests.Session for persistent headers
        self._http = requests.Session()
        if api_key:
            self._http.headers["X-API-Key"] = api_key

    # ── Session Management ──────────────────────────────────────────────

    def create_session(
        self, mode: str = "free_play", player_name: str = "student", config: dict = None
    ) -> dict:
        """
        Create a new game session.

        Args:
            mode: Game mode — "free_play", "target_practice", "time_trial",
                  "exploration", "anomaly_arena", or "competition"
            player_name: Display name for the agent
            config: Optional configuration overrides. Notable keys:
                - terrain_seed (int): pin the terrain layout to a specific seed
                  for reproducibility. Omit for a fresh random terrain each run.

        Returns:
            Session info dict with session_id, mode, status, browser_url

        Example:
            client.create_session(mode="time_trial", config={"terrain_seed": 42})
        """
        payload = {"mode": mode, "player_name": player_name, "config": config or {}}
        resp = self._http.post(f"{self.server_url}/api/session", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        browser_url = data.get("browser_url", f"http://localhost:5173/?session={self.session_id}")
        print(f"Session created: {self.session_id} (mode: {mode})")
        print(f"Open this URL in your browser: {browser_url}")
        return data

    def get_state(self) -> dict:
        """Get the full game state (agent position, score, wind, etc.)."""
        self._check_session()
        resp = self._http.get(f"{self.server_url}/api/session/{self.session_id}/state")
        resp.raise_for_status()
        return resp.json()

    def configure(self, **kwargs) -> dict:
        """
        Configure session settings.

        Keyword Args:
            wind_enabled (bool): Enable/disable wind
            terrain_variation (bool): Enable/disable terrain elevation
            fixed_force (float|None): Lock projectile force to a value, or None for variable
            obstacles_enabled (bool): Enable/disable arena obstacles
            sim_speed (float): Simulation speed multiplier (0.5, 1, 2, 4, or 8)
            terrain_seed (int|None): Re-roll terrain to a reproducible layout.
                Pass an integer to pin the terrain; pass None for a fresh random one.
            sensor_config (dict): Sensor configuration overrides

        Returns:
            Updated configuration
        """
        self._check_session()
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/configure", json=kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def delete_session(self):
        """Delete the current session and clean up."""
        if self.session_id:
            self._http.delete(f"{self.server_url}/api/session/{self.session_id}")
            self.session_id = None
        self.disconnect_ws()

    # ── Sensors ─────────────────────────────────────────────────────────

    def get_sensors(self) -> dict:
        """
        Get current sensor readings.

        Returns:
            Dict with position, velocity, heading, speed, ground, rays,
            navigation, wind, timestamp
        """
        self._check_session()
        resp = self._http.get(f"{self.server_url}/api/session/{self.session_id}/sensors")
        resp.raise_for_status()
        return resp.json()

    def get_ground_grid(self) -> dict:
        """
        Get 5x5 terrain grid centered on the agent.

        Returns:
            Dict with grid_size, cell_size, cells (5x5 array of terrain info)
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/grid"
        )
        resp.raise_for_status()
        return resp.json()

    def get_grid_observation(self) -> np.ndarray:
        """
        Get 32x32x4 grid observation for CNN input.

        Returns:
            numpy array of shape (4, 32, 32) — channels-first for PyTorch.
            Channel 0: terrain type, Channel 1: elevation,
            Channel 2: obstacles, Channel 3: navigation gradient
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/grid32"
        )
        resp.raise_for_status()
        data = resp.json()
        grid = np.array(data["grid"]["data"])  # (32, 32, 4)
        return grid.transpose(2, 0, 1)  # → (4, 32, 32) channels-first

    def cache_world_map(self, force: bool = False) -> dict:
        """
        Download the static world-map snapshot for local grid computation.

        Pulls the full 100x100 terrain grid (terrain IDs, elevations,
        obstacles) and checkpoint positions once per session. Subsequent
        get_grid_local() calls build the 32x32 heading-aligned grid from this
        cache without any server round-trip — sub-millisecond per call.

        Args:
            force: Re-download even if a snapshot is already cached.

        Returns:
            The cached snapshot dict (also stored on self._world_map).

        Caveats:
            * The snapshot is *static*. Modes that mutate terrain at runtime
              (anomaly_arena, or any call to configure(terrain_seed=...)) will
              invalidate it. In anomaly_arena, keep using get_grid_observation()
              instead. After reconfiguring the seed, call cache_world_map(force=True).
            * Channels 0-2 are derived from the snapshot; channel 3
              (nav gradient) is recomputed each tick from live position +
              checkpoint index, so it stays correct as the agent advances.
        """
        if self._world_map is not None and not force:
            return self._world_map
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/world_map"
        )
        resp.raise_for_status()
        raw = resp.json()
        gs = int(raw["grid_size"])
        cache = {
            "resolution": float(raw["resolution"]),
            "world_size": float(raw["world_size"]),
            "grid_size": gs,
            "x_min": float(raw["x_min"]),
            "z_min": float(raw["z_min"]),
            "terrain_ids": np.asarray(raw["terrain_ids"], dtype=np.float32).reshape(gs, gs),
            "elevations": np.asarray(raw["elevations"], dtype=np.float32).reshape(gs, gs),
            "obstacles": np.asarray(raw["obstacles"], dtype=np.float32).reshape(gs, gs),
            "checkpoints": list(raw.get("checkpoints", []) or []),
            "world_version": int(raw.get("world_version", 0)),
        }
        self._world_map = cache
        return cache

    def get_grid_local(self) -> np.ndarray:
        """
        Build the 32x32x4 CNN grid locally from the cached world snapshot.

        Identical shape and normalization to get_grid_observation(), but
        computed from cached arrays + the latest WS state — no per-tick
        network round-trip. Designed for 20 Hz inference loops where the
        REST/WS round-trip of get_grid_observation() is too slow.

        Returns:
            numpy array of shape (4, 32, 32), channels-first.
            Channel 0: terrain type, Channel 1: elevation,
            Channel 2: obstacles, Channel 3: navigation gradient.

        Requires:
            * connect_ws() called and at least one state broadcast received,
              so self._latest_state holds the live position + heading.
            * cache_world_map() called once (auto-invoked on first use).

        Caveat: snapshot is static — see cache_world_map() docstring. In
        anomaly_arena mode, fall back to get_grid_observation().
        """
        if self._world_map is None:
            self.cache_world_map()
        cache = self._world_map

        with self._state_lock:
            state = self._latest_state
        if state is None:
            raise RuntimeError(
                "no live state yet — call connect_ws() and wait one tick"
            )

        pos = state.get("position") or {}
        px = float(pos.get("x", 0.0))
        pz = float(pos.get("z", 0.0))
        heading = float(state.get("heading", 0.0))

        sensors = state.get("sensors") or {}
        nav = sensors.get("navigation") or {}
        cps_completed = int(nav.get("checkpoints_completed", 0))

        # 32×32 heading-aligned cell centers in world space — mirror of
        # SensorSystem.getGrid32x32 in src/agents/SensorSystem.ts.
        GRID = 32
        CELL = 2.0
        HALF = (GRID - 1) / 2.0  # 15.5
        cols = np.arange(GRID, dtype=np.float32)
        rows = np.arange(GRID, dtype=np.float32)
        cc, rr = np.meshgrid(cols, rows)
        local_x = (cc - HALF) * CELL
        local_z = (rr - HALF) * CELL
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        world_x = px + local_x * cos_h - local_z * sin_h
        world_z = pz + local_x * sin_h + local_z * cos_h

        # Nearest-neighbor sample of the 100×100 snapshot. The server's
        # internal terrain grid resolution (2 units/cell) matches grid32's
        # cell size exactly, so this is exact, not approximate.
        res = cache["resolution"]
        gs = cache["grid_size"]
        ix = np.floor((world_x - cache["x_min"]) / res).astype(np.int32)
        iz = np.floor((world_z - cache["z_min"]) / res).astype(np.int32)
        ix_c = np.clip(ix, 0, gs - 1)
        iz_c = np.clip(iz, 0, gs - 1)

        terrain = cache["terrain_ids"][iz_c, ix_c]
        elev = cache["elevations"][iz_c, ix_c]
        obs = cache["obstacles"][iz_c, ix_c]

        # Cells outside ±100 are obstacles, matching the server's bounds check.
        oob = (np.abs(world_x) > 100) | (np.abs(world_z) > 100)
        obs = np.where(oob, 1.0, obs).astype(np.float32)

        ch0 = (terrain / 6.0).astype(np.float32)
        MIN_H = -4.0
        MAX_H = 4.0
        ch1 = np.clip((elev - MIN_H) / (MAX_H - MIN_H), 0.0, 1.0).astype(np.float32)
        ch2 = obs

        checkpoints = cache["checkpoints"]
        if checkpoints:
            target = checkpoints[cps_completed % len(checkpoints)]["position"]
            tx = float(target.get("x", 0.0))
            tz = float(target.get("z", 0.0))
            dx = world_x - tx
            dz = world_z - tz
            dist = np.sqrt(dx * dx + dz * dz)
            ch3 = np.clip(1.0 - dist / 200.0, 0.0, 1.0).astype(np.float32)
        else:
            ch3 = np.zeros_like(ch0, dtype=np.float32)

        return np.stack([ch0, ch1, ch2, ch3], axis=0)

    def get_sensor_history(self, count: int = 100) -> list:
        """
        Get recent sensor history (rolling buffer).

        Args:
            count: Number of recent readings to retrieve (max 100)

        Returns:
            List of sensor readings with timestamps
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/sensors/history",
            params={"count": count},
        )
        resp.raise_for_status()
        return resp.json()["history"]

    # ── Projectiles ─────────────────────────────────────────────────────

    def fire_projectile(
        self, angle: float = 45, force: float = 50, yaw_offset: float = 0
    ) -> dict:
        """
        Fire a single projectile and wait for it to land.

        Args:
            angle: Launch angle in degrees (0-90)
            force: Launch force (0-100)
            yaw_offset: Horizontal offset in degrees (-45 to +45)

        Returns:
            Dict with launch/landing positions, distance, flight_time,
            trajectory, target hit info
        """
        self._check_session()
        payload = {"angle": angle, "force": force, "yaw_offset": yaw_offset}
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/fire",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def fire_batch(self, projectiles: list) -> list:
        """
        Fire multiple projectiles sequentially and wait for all to land.

        Args:
            projectiles: List of dicts, each with keys:
                angle (float), force (float), yaw_offset (float, optional)
                Example: [{"angle": 30, "force": 50}, {"angle": 45, "force": 70}]

        Returns:
            List of result dicts (same format as fire_projectile)
        """
        self._check_session()
        payload = {"projectiles": projectiles}
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/fire/batch",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def get_targets(self) -> list:
        """
        Get target positions (target practice mode).

        Returns:
            List of target dicts with id, position, radius, points
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/targets"
        )
        resp.raise_for_status()
        return resp.json()["targets"]

    def get_wind(self) -> dict:
        """
        Get current wind vector and metadata.

        Returns:
            Dict with wind vector, magnitude, direction_degrees, next_change_in
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/wind"
        )
        resp.raise_for_status()
        return resp.json()

    # ── Agent Control ───────────────────────────────────────────────────

    def send_control(self, throttle: float = 0, steering: float = 0) -> dict:
        """
        Send a control command via REST (low-frequency use).

        For 20Hz control loops, use connect_ws() and send_control_ws() instead.

        Args:
            throttle: Forward/backward force [-1, 1]
            steering: Left/right steering [-1, 1]

        Returns:
            Dict with applied status and timestamp
        """
        self._check_session()
        payload = {"throttle": throttle, "steering": steering}
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/control", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    # ── WebSocket (Real-time Control) ───────────────────────────────────

    def connect_ws(self, on_state=None):
        """
        Connect via WebSocket for real-time 20Hz control.

        Args:
            on_state: Optional callback function called with each state update.
                      Signature: on_state(state_dict) -> None
        """
        self._check_session()
        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        ws_url = f"{ws_scheme}://{self.server_url.split('//')[1]}/ws?session_id={self.session_id}"
        if self.api_key:
            ws_url += f"&api_key={self.api_key}"

        if on_state:
            self._callbacks["state"] = on_state

        def on_message(ws, message):
            data = json.loads(message)
            if data.get("type") == "state":
                with self._state_lock:
                    self._latest_state = data
                if "state" in self._callbacks:
                    self._callbacks["state"](data)
            elif data.get("type") == "pong":
                pass
            else:
                # Event notifications (checkpoint_reached, lap_complete, etc.)
                event_type = data.get("type")
                if event_type in self._callbacks:
                    self._callbacks[event_type](data)

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status, close_msg):
            print("WebSocket disconnected")

        def on_open(ws):
            print(f"WebSocket connected (session: {self.session_id})")

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        time.sleep(0.5)  # Wait for connection

    def send_control_ws(self, throttle: float = 0, steering: float = 0):
        """
        Send a control command via WebSocket (for 20Hz loops).

        Must call connect_ws() first.

        Args:
            throttle: Forward/backward force [-1, 1]
            steering: Left/right steering [-1, 1]
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected. Call connect_ws() first.")
        msg = json.dumps(
            {
                "type": "control",
                "session_id": self.session_id,
                "throttle": float(np.clip(throttle, -1, 1)),
                "steering": float(np.clip(steering, -1, 1)),
            }
        )
        self._ws.send(msg)

    def get_latest_state(self) -> dict:
        """
        Get the most recent state received via WebSocket.

        Returns:
            Latest state dict, or None if no state received yet
        """
        with self._state_lock:
            return self._latest_state

    def on_event(self, event_type: str, callback):
        """
        Register a callback for a specific event type.

        Args:
            event_type: Event name — "checkpoint_reached", "lap_complete",
                        "projectile_landed", "anomaly_start", "anomaly_end",
                        "round_start", "round_end", "competition_end"
            callback: Function called with the event data dict
        """
        self._callbacks[event_type] = callback

    def disconnect_ws(self):
        """Disconnect the WebSocket."""
        if self._ws:
            self._ws.close()
            self._ws = None

    # ── Recording (Behavioral Cloning) ──────────────────────────────────

    def start_recording(
        self, sample_rate: int = 20, include_grid: bool = False
    ) -> dict:
        """
        Start recording (state, action) pairs for behavioral cloning.

        Args:
            sample_rate: Samples per second (default: 20)
            include_grid: When True, also capture the 32x32x4 terrain grid per
                sample. Useful for CNN training. Adds ~5 MB per minute of
                recording at 20 Hz — fine over localhost; expect a slower
                download for long captures.

        Returns:
            Confirmation dict
        """
        self._check_session()
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/recording/start",
            json={"sample_rate": sample_rate, "include_grid": include_grid},
        )
        resp.raise_for_status()
        return resp.json()

    def stop_recording(self) -> dict:
        """Stop the current recording."""
        self._check_session()
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/recording/stop"
        )
        resp.raise_for_status()
        return resp.json()

    def get_recording(self) -> dict:
        """
        Download the recorded demonstration data.

        Returns:
            Dict with recording_id, duration, sample_rate, total_samples,
            and samples list (each with timestamp, state, action)
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/recording"
        )
        resp.raise_for_status()
        return resp.json()

    def get_recording_as_arrays(self) -> tuple:
        """
        Download recording and convert to numpy arrays for training.

        Returns:
            (states, actions) where:
            - states: np.ndarray of shape (N, 12) — normalized sensor features
            - actions: np.ndarray of shape (N, 2) — [throttle, steering]
        """
        recording = self.get_recording()
        states = []
        actions = []
        for sample in recording["samples"]:
            s = sample["state"]
            state_vec = [
                s["speed"],
                s["heading_error"],
                s["checkpoint_distance"],
                *s["rays"],
                s["ground_friction"],
            ]
            action_vec = [sample["action"]["throttle"], sample["action"]["steering"]]
            states.append(state_vec)
            actions.append(action_vec)
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

    def get_recording_positions(self) -> np.ndarray:
        """
        Download the recorded agent positions as a numpy array.

        Useful for plotting a high-Hz training path (denser than the 1 Hz
        polled track from `eval.run_policy`).

        Returns:
            np.ndarray of shape (N, 2) — columns are [x, z].
        """
        recording = self.get_recording()
        positions = []
        for sample in recording["samples"]:
            s = sample["state"]
            positions.append([s.get("position_x", 0.0), s.get("position_z", 0.0)])
        return np.array(positions, dtype=np.float32)

    def get_recording_with_grid(self) -> tuple:
        """
        Download a grid-enabled recording and return arrays for CNN training.

        Call `start_recording(..., include_grid=True)` first; without that
        flag the per-sample grid will be missing and this method raises.

        Returns:
            (states, actions, grid_stack) where:
            - states: np.ndarray of shape (N, 12) — same 12-feat vector as
              `get_recording_as_arrays`
            - actions: np.ndarray of shape (N, 2) — [throttle, steering]
            - grid_stack: np.ndarray of shape (N, 32, 32, 4) — per-sample
              terrain grid (heading-aligned). Bandwidth: ~5 MB per minute at
              20 Hz; expect a slower download for long captures.
        """
        recording = self.get_recording()
        states = []
        actions = []
        grids = []
        for sample in recording["samples"]:
            s = sample["state"]
            if "grid32" not in s:
                raise RuntimeError(
                    "recording has no grid32 samples — start with "
                    "start_recording(include_grid=True)"
                )
            state_vec = [
                s["speed"],
                s["heading_error"],
                s["checkpoint_distance"],
                *s["rays"],
                s["ground_friction"],
            ]
            action_vec = [sample["action"]["throttle"], sample["action"]["steering"]]
            states.append(state_vec)
            actions.append(action_vec)
            grids.append(s["grid32"])
        grid_stack = np.array(grids, dtype=np.float32).reshape(-1, 32, 32, 4)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            grid_stack,
        )

    # ── Map & Exploration ───────────────────────────────────────────────

    def get_explored_map(self) -> dict:
        """
        Get exploration progress (exploration mode).

        Returns:
            Dict with explored_percentage, total_cells, explored_cells, grid
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/map/explored"
        )
        resp.raise_for_status()
        return resp.json()

    def get_terrain_ground_truth(self) -> list:
        """
        Get terrain type labels for the full map (ground truth).
        Use this AFTER unsupervised analysis to evaluate your clustering.

        Returns:
            List of dicts with x, z, terrain_type, terrain_name
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/map/terrain"
        )
        resp.raise_for_status()
        return resp.json()["samples"]

    # ── Anomaly System ──────────────────────────────────────────────────

    def configure_anomalies(
        self,
        enabled: bool = True,
        malfunction_rate: float = 0.1,
        terrain_anomaly_rate: float = 0.05,
        malfunction_types: list = None,
        duration_range: tuple = (10, 50),
    ) -> dict:
        """
        Configure anomaly injection (anomaly arena mode).

        Args:
            enabled: Turn anomaly injection on/off
            malfunction_rate: Probability per second of new malfunction
            terrain_anomaly_rate: Fraction of terrain cells with wrong physics
            malfunction_types: List from ["steering_invert", "throttle_scale", "random_jitter"]
            duration_range: (min_ticks, max_ticks) for malfunction duration

        Returns:
            Configuration confirmation
        """
        self._check_session()
        payload = {
            "enabled": enabled,
            "agent_malfunction_rate": malfunction_rate,
            "terrain_anomaly_rate": terrain_anomaly_rate,
            "malfunction_types": malfunction_types
            or ["steering_invert", "throttle_scale", "random_jitter"],
            "malfunction_duration_range": list(duration_range),
        }
        resp = self._http.post(
            f"{self.server_url}/api/session/{self.session_id}/anomalies/configure",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def get_anomaly_labels(self) -> list:
        """
        Get ground truth anomaly labels (for evaluation after detection).

        Returns:
            List of dicts with timestamp, anomaly (bool), type
        """
        self._check_session()
        resp = self._http.get(
            f"{self.server_url}/api/session/{self.session_id}/anomalies/labels"
        )
        resp.raise_for_status()
        return resp.json()["labels"]

    # ── Competition ─────────────────────────────────────────────────────

    def join_competition(self, student_id: str, agent_name: str) -> dict:
        """
        Join a multiplayer competition room.

        Args:
            student_id: Your student identifier
            agent_name: Display name for your agent

        Returns:
            Dict with room_id, ws_url, status, player count
        """
        payload = {"student_id": student_id, "agent_name": agent_name}
        resp = self._http.post(
            f"{self.server_url}/api/competition/join", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"Joined competition room: {data['room_id']} ({data['players']} players)")
        return data

    def get_competition_state(self) -> dict:
        """
        Get current competition state (round, agents, scores).

        Returns:
            Dict with round info, agent positions, and scores
        """
        resp = self._http.get(f"{self.server_url}/api/competition/state")
        resp.raise_for_status()
        return resp.json()

    def get_leaderboard(self) -> dict:
        """
        Get current competition leaderboard.

        Returns:
            Dict with rounds_completed, total_rounds, standings
        """
        resp = self._http.get(f"{self.server_url}/api/competition/leaderboard")
        resp.raise_for_status()
        return resp.json()

    def connect_competition_ws(self, student_id: str, agent_name: str, on_state=None):
        """
        Connect to the competition WebSocket for multiplayer control.

        Args:
            student_id: Your student identifier
            agent_name: Display name for your agent
            on_state: Optional callback for competition state updates
        """
        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        ws_url = (
            f"{ws_scheme}://{self.server_url.split('//')[1]}/ws/competition"
            f"?student_id={student_id}&agent_name={agent_name}"
        )
        if self.api_key:
            ws_url += f"&api_key={self.api_key}"

        if on_state:
            self._callbacks["competition_state"] = on_state

        def on_message(ws, message):
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "state":
                with self._state_lock:
                    self._latest_state = data
                if "state" in self._callbacks:
                    self._callbacks["state"](data)
            elif msg_type == "competition_state":
                if "competition_state" in self._callbacks:
                    self._callbacks["competition_state"](data)
            elif msg_type in self._callbacks:
                self._callbacks[msg_type](data)

        def on_error(ws, error):
            print(f"Competition WebSocket error: {error}")

        def on_close(ws, close_status, close_msg):
            print("Competition WebSocket disconnected")

        def on_open(ws):
            print(f"Competition WebSocket connected ({student_id}: {agent_name})")

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        time.sleep(0.5)

    # ── Utilities ───────────────────────────────────────────────────────

    def collect_sensor_data(self, n_samples: int, interval: float = 0.2) -> list:
        """
        Collect multiple sensor readings while the agent moves.

        Useful for terrain classification data collection.

        Args:
            n_samples: Number of sensor readings to collect
            interval: Seconds between readings

        Returns:
            List of sensor reading dicts
        """
        self._check_session()
        readings = []
        for i in range(n_samples):
            reading = self.get_sensors()
            readings.append(reading)
            if i < n_samples - 1:
                time.sleep(interval)
            if (i + 1) % 50 == 0:
                print(f"Collected {i + 1}/{n_samples} samples")
        print(f"Collection complete: {len(readings)} samples")
        return readings

    def run_control_loop(self, policy_fn, duration: float = 60, hz: float = 20):
        """
        Run a control loop using a policy function.

        The policy function receives the current state and returns (throttle, steering).
        Uses WebSocket for real-time control at the specified frequency.

        Args:
            policy_fn: Function with signature (state_dict) -> (throttle, steering)
            duration: How long to run in seconds
            hz: Control frequency in Hz (default: 20)

        Example:
            def my_policy(state):
                # Your ML model here
                throttle = model.predict(state)[0]
                steering = model.predict(state)[1]
                return throttle, steering

            client.run_control_loop(my_policy, duration=60)
        """
        self._check_session()

        if not self._ws:
            self.connect_ws()
            time.sleep(0.5)

        interval = 1.0 / hz
        start_time = time.time()
        steps = 0

        print(f"Running control loop at {hz}Hz for {duration}s...")
        try:
            while time.time() - start_time < duration:
                state = self.get_latest_state()
                if state is not None:
                    throttle, steering = policy_fn(state)
                    self.send_control_ws(throttle, steering)
                    steps += 1
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Control loop interrupted")

        elapsed = time.time() - start_time
        print(f"Control loop finished: {steps} steps in {elapsed:.1f}s ({steps/elapsed:.1f} Hz)")

    def _check_session(self):
        """Verify a session exists."""
        if not self.session_id:
            raise RuntimeError(
                "No active session. Call create_session() first."
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect_ws()
        self.delete_session()

    def __repr__(self):
        status = f"session={self.session_id}" if self.session_id else "no session"
        ws = "ws=connected" if self._ws else "ws=disconnected"
        return f"GameClient({self.server_url}, {status}, {ws})"
