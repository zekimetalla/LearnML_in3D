# Instructor TODO — platform changes the project assumes

These are server-side / platform-side changes the project design assumes will be in place before students start. Listed in dependency order.

## Before students start iterating

- [x] **Honor the `seed` config in `create_session`.** `server/routes/session.ts` accepts both `terrain_seed` and `seed`, appends `&seed=N` to the browser URL, and `src/main.ts` feeds it into `TerrainGenerator` (Mulberry32 PRNG). `CheckpointSystem` uses a fixed `Math.sin(angle*3)*20` formula so checkpoint placement is identical across runs at any seed — same-seed reproducibility is what benchmarks need; all seeds share the same checkpoint layout by design.
- [x] **Verify the WS state broadcast includes `sensors.navigation.checkpoints_completed`.** Emitted in `src/main.ts` and consumed by `drive2win/eval.py`.
- [x] **Verify `state.position` is in the WS broadcast.** Emitted in `src/main.ts` and consumed by `drive2win/eval.py`.
- [x] **Add a `crashed` event or position-reset signal to the WS state.** `Agent.getRespawnCount()` is exposed as `sensors.navigation.respawn_count` in the WS broadcast. `eval.py` now prefers the counter-diff and only falls back to the old `dx² + dz² > 25` heuristic for older browser builds.

## Recording — needed for the path overlay and for any CNN iteration

- [x] **`RecordingSystem.captureSample()` should also record `position.x, position.z`.** Captured as `state.position_x` / `state.position_z` per sample; exposed via the new `client.get_recording_positions() -> np.ndarray` helper. Kept separate from the 12-feat BC training vector to preserve that contract.
- [x] **(Optional, for students who try a CNN.)** Pass `include_grid=True` to `client.start_recording(...)` to capture the 32×32×4 terrain grid per sample (`state.grid32`). Pull the data with `client.get_recording_with_grid()`, which returns `(states, actions, grid_stack)` with `grid_stack.shape == (N, 32, 32, 4)`. Bandwidth ≈ 5 MB per minute at 20 Hz.

## Before the tournament

- [ ] **Implement / verify the live tournament format the project assumes:**
  - 5 rounds × 5 minutes
  - terrain seed changes per round (use `seed = base + round_idx` or similar)
  - 3 rounds without obstacles, 2 with
  - pass / fail bar = ≥ 1 full lap completed in any round
  - ranking = total checkpoints across all 5 rounds
- [ ] **Capacity test the live arena for 20 simultaneous bots × 5 rounds.** Verify a disconnecting client doesn't take the room down. Verify scoring aggregates correctly across rounds.
- [ ] **Confirm `auth.ts` API-key flow works for 20+ concurrent clients hitting the same room.**
- [ ] **Decide and publish the deadline contract:** when does a student's agent need to be live and connected? What happens if they reconnect mid-round?

## Optional but valuable

- [ ] A `--fast` flag for `benchmark.py` that bumps `sim_speed` to 4× so 5 runs of 60 s wall-clock take ~75 s instead of 5 min. Useful during iteration.
- [ ] A separate **evaluation seed** the students cannot read — keeps the leaderboard fair if the published `seed=42` accidentally becomes a target for overfitting. (The 5-round terrain rotation already achieves much of this, but a hidden eval seed is a useful belt-and-suspenders.)
