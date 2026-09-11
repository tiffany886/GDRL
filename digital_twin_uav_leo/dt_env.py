"""TwinnedWorld: a digital twin over the physical UavLeoEnv.

Syncs a full state snapshot every ``tau`` slots (or on threshold-triggered
resync in adaptive mode); between syncs the twin extrapolates with a
predictor. Decision policies read ``get_obs()`` (twin state, but UAV position
kept at the physical truth, since the UAV knows its own position).
"""
import numpy as np

from .predictors import predict
from .adaptive_sync import estimate_error, should_resync

_PRED_KEYS = ("user_pos", "user_vel", "hotspot_pos", "hotspot_vel",
              "task_bits", "cycles_per_bit")


class TwinnedWorld:
    def __init__(self, env, tau=3, predictor="linear", eps=0.0, rng=None,
                 sync_mode="fixed", probe_m=2, probe_n=3, delta=10.0,
                 task_mode="per_user", sync_delay=0, sync_loss=0.0,
                 chan_rng=None):
        self.env = env
        self.cfg = env.config
        self.tau = int(tau)
        self.predictor = predictor
        self.eps = float(eps)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.sync_mode = sync_mode
        self.probe_m = int(probe_m)
        self.probe_n = int(probe_n)
        self.delta = float(delta)
        self.task_mode = task_mode
        self.since_sync = 0
        self.sync_count = 0
        # --- imperfect refresh channel (delay and/or loss) ---
        self.sync_delay = int(sync_delay)
        self.sync_loss = float(sync_loss)
        self.chan_rng = chan_rng if chan_rng is not None else np.random.default_rng(12345)
        self._pending = []          # (arrival_t, obs)
        self.requested_syncs = 0
        self.dropped_syncs = 0
        self.delivered_syncs = 0
        self._t = int(getattr(env, "t", 0))
        self._probe_counter = 0
        self._obs_physical = None
        self._pred = None

    @property
    def has_pending(self):
        return len(self._pending) > 0

    def sync(self, obs, immediate=False):
        """Request a refresh.

        With ``sync_delay == 0`` and ``sync_loss == 0`` this installs the
        snapshot immediately (original behaviour).  Otherwise the request may
        be dropped (probability ``sync_loss``) or queued for arrival
        ``sync_delay`` slots later, modelling a finite-bandwidth control plane
        rather than a zero-latency, lossless one.
        """
        self.requested_syncs += 1
        if immediate:
            self._install(obs)
            return
        if self.sync_loss > 0 and self.chan_rng.random() < self.sync_loss:
            self.dropped_syncs += 1
            return
        if self.sync_delay > 0:
            self._pending.append((self._t + self.sync_delay, obs))
            return
        self._install(obs)

    def _deliver_pending(self):
        if not self._pending:
            return
        due = [o for (a, o) in self._pending if a <= self._t]
        self._pending = [(a, o) for (a, o) in self._pending if a > self._t]
        if due:
            self._install(due[-1])
            self.delivered_syncs += 1

    def _install(self, obs):
        self._obs_physical = obs
        self._pred = {k: (obs[k].copy() if hasattr(obs[k], "copy") else obs[k])
                      for k in _PRED_KEYS}
        self._t = int(getattr(self.env, "t", 0))
        self.since_sync = 0
        self.sync_count += 1

    def step_forward(self, dt=1.0):
        self._t += 1
        self._deliver_pending()
        self._pred = predict(self.predictor, self._pred, dt, self.cfg,
                             self.rng, self.eps, t=self._t)
        self.since_sync += 1

    def get_obs(self):
        obs = dict(self._obs_physical)
        for k in _PRED_KEYS:
            obs[k] = self._pred[k]
        # UAV 位置/电量保留物理真值（自定位已知）
        obs["uav_pos"] = self._obs_physical["uav_pos"]
        obs["uav_battery"] = self._obs_physical["uav_battery"]
        return obs

    def need_resync(self):
        if self.sync_mode == "fixed":
            return self.since_sync >= self.tau
        return False

    def _maybe_probe_and_resync(self, obs_physical):
        self._obs_physical = obs_physical
        if self.sync_mode != "adaptive":
            return False
        self._probe_counter += 1
        if self._probe_counter < self.probe_m:
            return False
        self._probe_counter = 0
        err = estimate_error(self.get_obs(), obs_physical, self.probe_n,
                             self.rng, cfg=self.cfg, task_mode=self.task_mode)
        if should_resync(err, self.delta):
            self.sync(obs_physical)
            return True
        return False
