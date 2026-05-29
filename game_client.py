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
import math
import time
import threading
import numpy as np
import requests
import websocket


# Ground-friction lookup keyed by terrain ID (mirror of TERRAIN_TYPES in
# shared/types.ts). Same values, same order: grass, dirt, sand, mud, ice,
# rock, pavement.
TERRAIN_FRICTION = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.4, 5: 1.2, 6: 1.1}

# Ray angles in degrees, ported verbatim from src/agents/SensorSystem.ts:
# RAYCAST_ANGLES. 0 = forward, increasing CCW relative to heading.
RAYCAST_ANGLES_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
RAYCAST_MAX_RANGE = 50.0


def _parse_world_map_payload(raw: dict) -> dict:
    """Convert a WorldMapSnapshot JSON payload into the numpy-backed dict
    used by _compute_grid_local() and _compute_rays_8().

    Mirrors GameClient.cache_world_map's old inline body so both the
    single-player client and RoomBot build the same cache structure.
    """
    gs = int(raw["grid_size"])
    return {
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


def _compute_grid_local(cache: dict, px: float, pz: float, heading: float, cps_completed: int) -> np.ndarray:
    """Build the 32×32×4 heading-aligned CNN grid from a cached snapshot.

    Same math as the original GameClient.get_grid_local — extracted as a free
    function so RoomBot can reuse it without holding a session.
    """
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

    res = cache["resolution"]
    gs = cache["grid_size"]
    ix = np.floor((world_x - cache["x_min"]) / res).astype(np.int32)
    iz = np.floor((world_z - cache["z_min"]) / res).astype(np.int32)
    ix_c = np.clip(ix, 0, gs - 1)
    iz_c = np.clip(iz, 0, gs - 1)

    terrain = cache["terrain_ids"][iz_c, ix_c]
    elev = cache["elevations"][iz_c, ix_c]
    obs = cache["obstacles"][iz_c, ix_c]

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


def _compute_rays_8(cache: dict, px: float, pz: float, heading: float,
                    max_dist: float = RAYCAST_MAX_RANGE) -> list:
    """8-direction obstacle raycast against the cached obstacle grid.

    Replica of SensorSystem.getRaycasts, but using the cached 100×100
    obstacle map instead of a Rapier physics world. Direction convention
    matches the TS source exactly:
        dirX = -sin(angleRad + heading)
        dirZ = -cos(angleRad + heading)
    so angle 0 == "forward" relative to the bot.

    Returns a list of 8 floats (world units), each in [0, max_dist].
    """
    res = float(cache["resolution"])
    gs = int(cache["grid_size"])
    x_min = float(cache["x_min"])
    z_min = float(cache["z_min"])
    obstacles = cache["obstacles"]

    # Step at quarter-cell granularity — well under the cell width so we
    # don't skip past thin obstacles. At res=2 that's 0.5 world units,
    # giving ~1% precision vs the 50-unit max range (the parity target
    # mentioned in the tournament plan).
    step = 0.5
    n_steps = int(max_dist / step) + 1

    distances = []
    for angle_deg in RAYCAST_ANGLES_DEG:
        angle_rad = (angle_deg * math.pi) / 180.0 + heading
        dir_x = -math.sin(angle_rad)
        dir_z = -math.cos(angle_rad)

        hit_dist = max_dist
        for k in range(1, n_steps + 1):
            d = k * step
            wx = px + dir_x * d
            wz = pz + dir_z * d
            if abs(wx) > 100 or abs(wz) > 100:
                hit_dist = d
                break
            ix = int(math.floor((wx - x_min) / res))
            iz = int(math.floor((wz - z_min) / res))
            if ix < 0 or ix >= gs or iz < 0 or iz >= gs:
                hit_dist = d
                break
            if obstacles[iz, ix] >= 0.5:
                hit_dist = d
                break
        distances.append(float(min(hit_dist, max_dist)))

    return distances


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
        cache = _parse_world_map_payload(resp.json())
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

        return _compute_grid_local(cache, px, pz, heading, cps_completed)

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


# ──────────────────────────────────────────────────────────────────────────
#  Tournament client — RoomBot
#
#  Connects to a multiplayer room's bot channel (/ws/room/bot). Different
#  lifecycle from GameClient: no HTTP session creation, no admin browser
#  ownership — just join a room by name, signal ready, drive when round_start
#  fires. Reuses _parse_world_map_payload, _compute_grid_local, and
#  _compute_rays_8 so the observation a controller sees in a tournament
#  matches what students trained on with GameClient.
# ──────────────────────────────────────────────────────────────────────────


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (heading) from a quaternion using THREE.js YXZ Euler
    order — matches Agent.getHeading() exactly. See src/agents/Agent.ts:100.
    """
    # YXZ extraction: heading = atan2(-m20, m22) where
    #   m20 = 2*(x*z - w*y), m22 = 1 - 2*(x*x + y*y)
    sy = 2.0 * (qw * qy - qx * qz)
    cy = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(sy, cy)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class RoomBot:
    """Tournament client. Plug a controller in, call run(), drive until the
    tournament ends.

    Example:
        from game_client import RoomBot

        def my_controller(obs):
            # obs is a dict — see TOURNAMENT.md for the full schema
            return 0.7, obs["navigation"]["heading_error"] * 0.4

        bot = RoomBot("https://ml.ferit.tech", room="demo", name="Alice",
                      api_key="mlsim_abc123")
        standings = bot.run(my_controller, hz=20.0)
        print(standings)
    """

    def __init__(
        self,
        server_url: str = "https://ml.ferit.tech",
        room: str = "main",
        name: str = "bot",
        api_key: str = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.room = room
        self.name = name
        self.api_key = api_key

        self._ws = None
        self._ws_thread = None
        self._stop = threading.Event()
        self._connected = threading.Event()

        # Latest message-derived state, written by the WS thread, read by the
        # control loop. All access through this lock.
        self._lock = threading.Lock()
        self._bot_key = None
        self._phase = "lobby"          # lobby/countdown/racing/round_end/finished
        self._round_index = 0
        self._latest_bots = []         # list of RoomBotState dicts
        self._latest_state_t = 0
        self._world_map = None
        self._world_map_round = -1
        self._standings = []
        self._tournament_done = False
        self._last_pos_for_speed = None  # (x, z, t) for speed estimation
        self._last_speed = 0.0

        # HTTP session for the world_map fetch.
        self._http = requests.Session()
        if api_key:
            self._http.headers["X-API-Key"] = api_key

    # ── public API ─────────────────────────────────────────────────────

    def run(self, controller, hz: float = 20.0) -> list:
        """Connect, ready up, drive at `hz` until tournament_end.

        Args:
            controller: Callable taking an `obs` dict and returning
                (throttle, steering) — both in [-1, 1].
            hz: Control tick rate.

        Returns:
            Final standings list (each entry is the TournamentStanding dict
            broadcast by the server).
        """
        if self._ws is None:
            self._connect()
        # Wait briefly for the bot_assigned + initial snapshot before the
        # tick loop runs — otherwise the first few ticks would have no state.
        self._connected.wait(timeout=5.0)

        period = 1.0 / hz
        dt_warn_threshold = 0.100  # 100 ms — log if controller is too slow
        avg_dt = period
        next_tick = time.time()
        try:
            while not self._stop.is_set():
                with self._lock:
                    done = self._tournament_done
                if done:
                    break
                t0 = time.time()
                obs = self._build_obs()
                if obs is not None:
                    try:
                        out = controller(obs)
                        throttle, steering = float(out[0]), float(out[1])
                    except Exception as e:
                        # Defensive: a buggy controller shouldn't crash the
                        # tournament — coast instead and keep going.
                        print(f"[RoomBot:{self.name}] controller error: {e}")
                        throttle, steering = 0.0, 0.0
                    # Only send control while a round is actually running.
                    if self._phase == "racing":
                        self._send_control(throttle, steering)
                # Frame budget + slowness warning
                dt = time.time() - t0
                avg_dt = 0.9 * avg_dt + 0.1 * dt
                if avg_dt > dt_warn_threshold:
                    print(f"[RoomBot:{self.name}] WARNING: controller avg dt {avg_dt*1000:.0f}ms "
                          f"exceeds budget — frames will drop")
                    avg_dt = period  # reset so warning isn't spammy
                next_tick += period
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.time()  # we're behind — resync
        except KeyboardInterrupt:
            print(f"[RoomBot:{self.name}] interrupted")
        finally:
            self.disconnect()
        return list(self._standings)

    def disconnect(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── WS lifecycle ───────────────────────────────────────────────────

    def _connect(self) -> None:
        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        host = self.server_url.split("//", 1)[1]
        ws_url = f"{ws_scheme}://{host}/ws/room/bot?room={self.room}&name={self.name}"
        if self.api_key:
            ws_url += f"&api_key={self.api_key}"

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def _on_open(self, ws):
        print(f"[RoomBot:{self.name}] connected to room '{self.room}' — signaling ready")
        try:
            ws.send(json.dumps({"type": "ready", "ready": True}))
        except Exception:
            pass

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        if t == "bot_assigned":
            with self._lock:
                self._bot_key = msg.get("bot_key")
                rs = msg.get("room_state") or {}
                self._phase = rs.get("phase", "lobby")
                self._round_index = int(rs.get("round_index", 0))
            print(f"[RoomBot:{self.name}] bot_key={self._bot_key}")
            self._connected.set()
        elif t == "round_start":
            ridx = int(msg.get("round_index", 0))
            with self._lock:
                self._phase = "racing"
                self._round_index = ridx
                self._last_pos_for_speed = None
                self._last_speed = 0.0
            print(f"[RoomBot:{self.name}] round_start idx={ridx} seed={msg.get('seed')} "
                  f"obstacles={msg.get('obstacles')}")
            # Fetch the world map for this round (terrain + obstacles).
            self._fetch_world_map(ridx)
        elif t == "state_update":
            bots = msg.get("bots") or []
            with self._lock:
                self._latest_bots = bots
                self._latest_state_t = int(msg.get("t", 0))
        elif t == "round_end":
            with self._lock:
                self._phase = "round_end"
            print(f"[RoomBot:{self.name}] round_end idx={msg.get('round_index')}")
        elif t == "tournament_end":
            standings = msg.get("standings") or []
            with self._lock:
                self._phase = "finished"
                self._standings = standings
                self._tournament_done = True
            print(f"[RoomBot:{self.name}] tournament_end")
            for r in standings:
                print(f"  #{r.get('rank')} {r.get('name')} cps={r.get('total_checkpoints')}")
        elif t == "error":
            code = msg.get("code")
            print(f"[RoomBot:{self.name}] error: {code} {msg.get('message')}")
            # Auth errors are unrecoverable — bail out so the user notices.
            if code in ("auth_failed", "unauthorized", "forbidden"):
                self._stop.set()

    def _on_error(self, ws, err):
        print(f"[RoomBot:{self.name}] ws error: {err}")

    def _on_close(self, ws, code, reason):
        print(f"[RoomBot:{self.name}] disconnected ({code} {reason})")
        # Releasing the connected event lets a still-waiting run() exit
        # instead of hanging forever if the server drops us pre-assignment.
        self._connected.set()
        self._stop.set()

    # ── observation building ───────────────────────────────────────────

    def _fetch_world_map(self, round_index: int) -> None:
        """GET /api/room/<room>/world_map and cache the parsed snapshot."""
        url = f"{self.server_url}/api/room/{self.room}/world_map"
        try:
            # First attempt may race the admin browser's race-start — retry a few times.
            for attempt in range(5):
                resp = self._http.get(url, timeout=3.0)
                if resp.status_code == 200:
                    cache = _parse_world_map_payload(resp.json())
                    with self._lock:
                        self._world_map = cache
                        self._world_map_round = round_index
                    print(f"[RoomBot:{self.name}] cached world_map for round {round_index}")
                    return
                if resp.status_code in (404, 504):
                    time.sleep(0.5)
                    continue
                resp.raise_for_status()
            print(f"[RoomBot:{self.name}] world_map fetch failed after retries (last status {resp.status_code})")
        except Exception as e:
            print(f"[RoomBot:{self.name}] world_map fetch error: {e}")

    def _self_state(self) -> dict:
        """Find this bot's RoomBotState in the latest state_update."""
        with self._lock:
            bot_key = self._bot_key
            bots = list(self._latest_bots)
        if not bot_key:
            return None
        for b in bots:
            if b.get("bot_key") == bot_key:
                return b
        return None

    def _other_bots(self) -> list:
        with self._lock:
            bot_key = self._bot_key
            bots = list(self._latest_bots)
        return [b for b in bots if b.get("bot_key") != bot_key]

    def _build_obs(self) -> dict:
        self_state = self._self_state()
        if self_state is None:
            return None
        pos = self_state.get("position") or {}
        rot = self_state.get("rotation") or {}
        px = float(pos.get("x", 0.0))
        py = float(pos.get("y", 0.0))
        pz = float(pos.get("z", 0.0))
        heading = _quat_to_yaw(
            float(rot.get("x", 0.0)),
            float(rot.get("y", 0.0)),
            float(rot.get("z", 0.0)),
            float(rot.get("w", 1.0)),
        )

        # Finite-difference speed estimator. Matches Agent.getSpeed() (which
        # ignores the Y component) by only using x/z deltas.
        now = time.time()
        with self._lock:
            last = self._last_pos_for_speed
        if last is None:
            speed = 0.0
        else:
            lx, lz, lt = last
            dt = max(now - lt, 1e-3)
            speed = math.sqrt((px - lx) ** 2 + (pz - lz) ** 2) / dt
            # Low-pass: room state arrives at ~20 Hz so per-tick deltas are
            # noisy. A light EMA keeps the value usable for BC inference.
            speed = 0.6 * speed + 0.4 * self._last_speed
        with self._lock:
            self._last_pos_for_speed = (px, pz, now)
            self._last_speed = speed

        cps_completed = int(self_state.get("checkpoints", 0))

        # Pull cached world map. If not yet fetched (first ticks of round 0),
        # serve zeroed sensors so the controller can still run.
        with self._lock:
            cache = self._world_map
            round_idx = self._round_index
            phase = self._phase

        if cache is not None:
            rays = _compute_rays_8(cache, px, pz, heading)
            ground_friction = self._lookup_ground_friction(cache, px, pz)
            grid32 = _compute_grid_local(cache, px, pz, heading, cps_completed)
            navigation = self._compute_navigation(cache, px, pz, heading, cps_completed)
        else:
            rays = [RAYCAST_MAX_RANGE] * 8
            ground_friction = 1.0
            grid32 = np.zeros((4, 32, 32), dtype=np.float32)
            navigation = {"distance": 0.0, "heading_error": 0.0, "checkpoint_index": 0}

        return {
            "position": {"x": px, "y": py, "z": pz},
            "heading": heading,
            "speed": float(speed),
            "rays": rays,
            "ground_friction": float(ground_friction),
            "grid32": grid32,
            "navigation": navigation,
            "checkpoints_passed": cps_completed,
            "round_index": round_idx,
            "race_phase": phase,
            "other_bots": self._other_bots(),
        }

    def _lookup_ground_friction(self, cache: dict, px: float, pz: float) -> float:
        res = float(cache["resolution"])
        gs = int(cache["grid_size"])
        ix = int(math.floor((px - float(cache["x_min"])) / res))
        iz = int(math.floor((pz - float(cache["z_min"])) / res))
        if ix < 0 or ix >= gs or iz < 0 or iz >= gs:
            return 1.0  # outside the world — friction undefined, default to grass
        tid = int(cache["terrain_ids"][iz, ix])
        return TERRAIN_FRICTION.get(tid, 1.0)

    def _compute_navigation(self, cache: dict, px: float, pz: float, heading: float, cps_completed: int) -> dict:
        """Mirror of CheckpointSystem.getNavigationInfo — same atan2(-dx,-dz)
        target-angle convention, same wrap to [-π, π]."""
        checkpoints = cache.get("checkpoints") or []
        if not checkpoints:
            return {"distance": 0.0, "heading_error": 0.0, "checkpoint_index": 0}
        idx = cps_completed % len(checkpoints)
        target = checkpoints[idx].get("position") or {}
        tx = float(target.get("x", 0.0))
        tz = float(target.get("z", 0.0))
        dx = tx - px
        dz = tz - pz
        distance = math.sqrt(dx * dx + dz * dz)
        target_angle = math.atan2(-dx, -dz)
        heading_error = _wrap_pi(target_angle - heading)
        return {
            "distance": float(distance),
            "heading_error": float(heading_error),
            "checkpoint_index": int(idx),
        }

    def _send_control(self, throttle: float, steering: float) -> None:
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({
                "type": "control",
                "throttle": float(np.clip(throttle, -1.0, 1.0)),
                "steering": float(np.clip(steering, -1.0, 1.0)),
            }))
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __repr__(self) -> str:
        return f"RoomBot({self.server_url}, room={self.room}, name={self.name})"

    def __repr__(self):
        status = f"session={self.session_id}" if self.session_id else "no session"
        ws = "ws=connected" if self._ws else "ws=disconnected"
        return f"GameClient({self.server_url}, {status}, {ws})"