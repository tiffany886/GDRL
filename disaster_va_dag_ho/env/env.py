"""VA-DAG-HO disaster environment (W1 scope).

Causal order implemented here is the locked one from the spec §6:

    arrivals(t) -> observation -> RL trajectory action -> move + flight energy
    -> rule-based association -> ScheduleDAG (new positions + current windows,
    W2 real implementation; W1 keeps a no-op stub) -> settle L/E/F -> reward
    -> t += 1

Energy: SoC ledger with flight cost every slot; comm/compute hooks exist and
are consumed by the scheduler from W2 on. No charging is modelled.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .channel import (
    flight_energy_joules,
    hard_cover_rates,
    terminal_uav_rate_matrix,
    uav_leo_capacity_matrix,
    uav_leo_channel_rate,
)
from .config import VAConfig
from .ephemeris import Ephemeris
from .mobility import Terminals
from .tasks import APP_DONE, APP_FAILED, APP_PENDING, AppRequest, ArrivalProcess

from ..scheduler.dag import generate_dag
from ..scheduler.schedule_dag import ScheduleDAG, SchedCtx
from ..agents.association import associate_terminals


class DisasterEnv:
    def __init__(self, cfg: VAConfig):
        self.cfg = cfg
        self.ephemeris = Ephemeris(cfg)
        self.arrival_process = None  # (re)built in reset
        self.reset(seed=cfg.seed)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.t = 0
        self.t_s = 0.0

        self.terminals = Terminals(cfg, self.rng)
        self.leo_sub = self._init_leo_subpoints()
        self.uav_pos = self._init_uav_pos()
        self.uav_soc = np.full(cfg.num_uavs, cfg.soc_init_j, dtype=float)

        self.arrival_process = ArrivalProcess(cfg, self.rng)
        self.scheduler = ScheduleDAG(cfg)
        self.scheduler.bind_ephemeris(self.ephemeris)
        self.apps: list[AppRequest] = []
        self.terminal_apps: list[list[int]] = [[] for _ in range(cfg.num_terminals)]
        self.next_app_id = 0
        self.assignment = np.full(cfg.num_terminals, -1, dtype=int)
        self.fail_history = deque(maxlen=max(cfg.obs_fail_window, 1))
        self._energy_before = 0.0

        self.metrics = {
            "flight_energy_j": 0.0,
            "comm_energy_j": 0.0,
            "compute_energy_j": 0.0,
            "energy_total_j": 0.0,
            "apps_arrived": 0,
            "apps_done": 0,
            "apps_failed": 0,
            "done_delay_sum_s": 0.0,
            "leo_invalid_attempts": 0,
            "leo_placements": 0,
            "dep_violations": 0,
            "latency_open_s": 0.0,
            "reward_sum": 0.0,
            "penalty_boundary": 0.0,
            "penalty_collision": 0.0,
            "penalty_low_battery": 0.0,
            "fail_uncovered": 0,               # v2 rule A: failures from lost coverage
            "unassociated_terminal_slots": 0.0,  # v2: sum over slots of assignment==-1 count
            "steps": 0,
        }
        return self.observation()

    @property
    def frac_unassociated(self) -> float:
        """Mean fraction of terminals with no UAV association per slot (v2 §1.3)."""
        steps = max(int(self.metrics["steps"]), 1)
        denom = steps * max(int(self.cfg.num_terminals), 1)
        return float(self.metrics["unassociated_terminal_slots"]) / denom

    def _init_leo_subpoints(self) -> np.ndarray:
        cfg = self.cfg
        c = cfg.area_size_m / 2.0
        offs = np.linspace(-cfg.area_size_m * 0.25, cfg.area_size_m * 0.25, cfg.num_leos)
        return np.stack([np.full(cfg.num_leos, c) + offs, np.full(cfg.num_leos, c)], axis=1)

    def _init_uav_pos(self) -> np.ndarray:
        cfg = self.cfg
        k = cfg.num_uavs
        area = cfg.area_size_m
        # spread UAVs evenly along x with a small jitter, y at mid-corridor
        if cfg.uav_init_xy is not None:
            flat = np.asarray(cfg.uav_init_xy, dtype=float).ravel()
            if flat.size != 2 * k:
                raise ValueError(
                    f"uav_init_xy needs {2 * k} coords for {k} UAVs, got {flat.size}")
            return np.clip(flat.reshape(k, 2), 0.0, area)
        xs = area * (2.0 * np.arange(k) + 1.0) / (2.0 * k)
        xs = xs + self.rng.uniform(-area * 0.03, area * 0.03, size=k)
        ys = np.full(k, area / 2.0) + self.rng.uniform(-area * 0.05, area * 0.05, size=k)
        pos = np.clip(np.stack([xs, ys], axis=1), 0.0, area)
        return pos

    # ------------------------------------------------------------------
    # observation (spec §6; flat per-UAV vector)
    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        cfg = self.cfg
        base = 8 + 2 * (cfg.num_uavs - 1) + cfg.num_leos + 1
        extra = 0
        if cfg.obs_terminal_features:
            cap = int(np.ceil(cfg.num_terminals / cfg.num_uavs))
            extra = cap * 4  # per assigned terminal: assoc/pending/ready/deadline
        return base + extra

    def observation(self) -> np.ndarray:
        cfg = self.cfg
        k = cfg.num_uavs
        area = cfg.area_size_m
        t_norm = min(self.t / max(cfg.horizon, 1), 1.0)
        recent_fail = 0.0
        if self.fail_history:
            recent_fail = float(np.mean(list(self.fail_history)[-cfg.obs_fail_window:]))
        backlog = self._backlog_per_uav()
        rows = []
        for kk in range(k):
            feats = [
                t_norm,
                self.uav_pos[kk, 0] / area,
                self.uav_pos[kk, 1] / area,
                (self.uav_soc[kk] / cfg.soc_init_j) if cfg.obs_energy else 0.0,
                min(backlog[kk] / 10.0, 1.0),
                self._avg_open_age_norm(kk),
            ]
            feats.extend(self._pending_centroid_offset(kk))
            for j in range(k):
                if j != kk:
                    feats.append((self.uav_pos[j, 0] - self.uav_pos[kk, 0]) / area)
                    feats.append((self.uav_pos[j, 1] - self.uav_pos[kk, 1]) / area)
            for l in range(cfg.num_leos):
                period = max(float(cfg.leo_window_period_s[l]) * float(cfg.leo_window_duty[l]), 1e-9)
                feats.append(min(self.ephemeris.remaining_s(l, self.t_s) / period, 1.0))
            feats.append(recent_fail)
            if cfg.obs_terminal_features:
                feats.extend(self._terminal_obs_block(kk))
            rows.append(feats)
        return np.asarray(rows, dtype=float)

    def _pending_centroid_offset(self, uav_k: int) -> list:
        """Normalized offset of the backlog centroid of own assigned terminals."""
        cfg = self.cfg
        area = cfg.area_size_m
        wsum = 0.0
        acc = np.zeros(2)
        for n in self._assigned_terminals(uav_k):
            for i in self.terminal_apps[n]:
                if self.apps[i].state == APP_PENDING:
                    acc += self.terminals.pos[n]
                    wsum += 1.0
        if wsum <= 0.0:
            return [0.0, 0.0]
        centroid = acc / wsum
        offset = (centroid - self.uav_pos[uav_k]) / area
        return [float(np.clip(offset[0], -1.0, 1.0)),
                float(np.clip(offset[1], -1.0, 1.0))]

    def _assigned_terminals(self, uav_k: int):
        terms = [n for n in range(self.cfg.num_terminals) if self.assignment[n] == uav_k]
        terms.sort()
        return terms

    def _terminal_obs_block(self, uav_k: int) -> list:
        cfg = self.cfg
        cap = int(np.ceil(cfg.num_terminals / cfg.num_uavs))
        terms = self._assigned_terminals(uav_k)
        block = []
        for s in range(cap):
            if s >= len(terms):
                block.extend([0.0, 0.0, 0.0, 0.0])
                continue
            n = terms[s]
            pend = [
                self.apps[i] for i in self.terminal_apps[n]
                if self.apps[i].state == APP_PENDING
            ]
            pending = len(pend)
            ready = 0.0
            rem = 0.0
            if pend:
                rem = float(np.clip(
                    min((a.deadline_s - self.t_s) for a in pend) / max(cfg.app_deadline_s, 1e-9),
                    0.0, 1.0,
                ))
                for a in pend:
                    if (a.app_id in self.scheduler.apps
                            and self.scheduler.ready_indices(a.app_id)):
                        ready = 1.0
                        break
            block += [
                1.0,
                min(pending / float(max(cfg.terminal_concurrency_max, 1)), 1.0),
                ready,
                rem,
            ]
        return block

    def _backlog_per_uav(self) -> np.ndarray:
        counts = np.zeros(self.cfg.num_uavs, dtype=int)
        for n, app_idxs in enumerate(self.terminal_apps):
            u = self.assignment[n]
            if u < 0:
                continue
            counts[u] += sum(self.apps[i].state == APP_PENDING for i in app_idxs)
        return counts

    def _avg_open_age_norm(self, uav_k: int) -> float:
        cfg = self.cfg
        ages = []
        for n in range(cfg.num_terminals):
            if self.assignment[n] != uav_k:
                continue
            for i in self.terminal_apps[n]:
                if self.apps[i].state == APP_PENDING:
                    ages.append(self.t_s - self.apps[i].arrival_s)
        if not ages:
            return 0.0
        return float(np.clip(np.mean(ages) / max(cfg.app_deadline_s, 1e-9), 0.0, 1.0))

    # ------------------------------------------------------------------
    # step (causal chain)
    # ------------------------------------------------------------------
    def step(self, uav_actions: np.ndarray | None = None,
             offload_decisions: np.ndarray | None = None):
        """Advance one slot.

        ``uav_actions``: (K, 2) horizontal velocity [m/s].
        ``offload_decisions`` (B4 flat mode only): (K, slots_per_uav) ints in
        {0=UAV, 1=LEO, 2=wait} selecting how each app with a ready node is
        treated this slot. ``None`` keeps the embedded ScheduleDAG decision
        rule (main method).
        """
        cfg = self.cfg
        self._offload_decisions = offload_decisions
        # 0) arrivals at current time
        pending_by_terminal = np.array(
            [sum(self.apps[i].state == APP_PENDING for i in idxs) for idxs in self.terminal_apps]
        )
        for n in self.arrival_process.sample_terminals(
            self.t_s, pending_by_terminal, self.terminals.pos
        ):
            weights = np.asarray(cfg.template_weights, dtype=float)
            weights = weights / weights.sum()
            template_id = int(self.rng.choice(len(weights), p=weights))
            app = AppRequest(
                app_id=self.next_app_id,
                terminal=n,
                arrival_s=self.t_s,
                deadline_s=self.t_s + cfg.app_deadline_s,
                template_id=template_id,
                dag=generate_dag(
                    template_id, self.rng,
                    cycles_scale=cfg.dag_cycles_scale,
                    bits_scale=cfg.dag_bits_scale,
                ),
            )
            self.apps.append(app)
            self.terminal_apps[n].append(app.app_id)
            self.next_app_id += 1
            self.metrics["apps_arrived"] += 1

        # delay proxy: count open apps at slot start (incl. just-arrived)
        n_open_during = sum(a.state == APP_PENDING for a in self.apps)

        # 1) gate + apply actions, move, pay flight energy
        self._energy_before = self.metrics["energy_total_j"]
        eff_actions, low_battery_flags = self._gate_actions(uav_actions)
        old_pos = self.uav_pos.copy()
        raw_desired = old_pos + eff_actions * cfg.slot_seconds
        out_x = (raw_desired[:, 0] < 0.0) | (raw_desired[:, 0] > cfg.area_size_m)
        out_y = (raw_desired[:, 1] < 0.0) | (raw_desired[:, 1] > cfg.area_size_m)
        n_boundary = int(np.count_nonzero(out_x | out_y))
        self.uav_pos = np.clip(raw_desired, 0.0, cfg.area_size_m)
        moved = np.linalg.norm(self.uav_pos - old_pos, axis=1)

        flight_j = np.array([flight_energy_joules(cfg, m) for m in moved])
        self._consume_uav_energy(flight_j, "flight")

        # 2) soft penalties
        collision_pairs = 0
        if cfg.num_uavs > 1:
            d2 = np.sum(
                (self.uav_pos[None, :, :] - self.uav_pos[:, None, :]) ** 2, axis=-1
            )
            np.fill_diagonal(d2, np.inf)
            collision_pairs = int(np.count_nonzero(np.sqrt(d2) < cfg.uav_min_sep_m)) // 2
        penalty_boundary = n_boundary * cfg.boundary_penalty
        penalty_collision = collision_pairs * cfg.collision_penalty
        penalty_low_battery = int(np.sum(low_battery_flags)) * cfg.low_battery_penalty

        # 3) rule-based association (post-move, spec §6: RL 不输出关联)
        self._associate()
        self.metrics["unassociated_terminal_slots"] += float(
            np.count_nonzero(self.assignment < 0)
        )
        # v2 rule A: pending apps whose terminal currently has no UAV
        # association cannot make progress; tag them so a later deadline
        # failure is counted as `fail_uncovered`.
        for app in self.apps:
            if app.state == APP_PENDING and self.assignment[app.terminal] < 0:
                app.suffered_uncovered = True

        # 4) scheduling (ScheduleDAG, spec §7)
        self._run_scheduling()

        # 5) settle L/E/F
        t_end = self.t_s + cfg.slot_seconds
        failed_this_step = self._settle_deadlines(t_end)
        n_failed = len(failed_this_step)
        self.metrics["apps_failed"] += n_failed
        self.metrics["fail_uncovered"] += sum(
            1 for app in failed_this_step if app.suffered_uncovered
        )
        if n_failed:
            self.fail_history.append(1)
        elif self.fail_history:
            self.fail_history.append(0)
        L = n_open_during * cfg.slot_seconds  # open-app time this slot
        self.metrics["latency_open_s"] += L

        # 6) reward
        e_this = self.metrics["energy_total_j"] - self._energy_before
        reward = -(
            cfg.alpha_latency * L
            + cfg.alpha_energy * e_this
            + cfg.alpha_fail * n_failed
        ) - (penalty_boundary + penalty_collision + penalty_low_battery)
        self.metrics["reward_sum"] += reward
        self.metrics["penalty_boundary"] += penalty_boundary
        self.metrics["penalty_collision"] += penalty_collision
        self.metrics["penalty_low_battery"] += penalty_low_battery
        self.metrics["steps"] += 1

        self.t += 1
        self.t_s = self.t * cfg.slot_seconds
        done = self.t >= cfg.horizon

        info = {
            "t": self.t,
            "L": L,
            "e_joules": e_this,
            "n_failed": n_failed,
            "reward": reward,
            "penalties": {
                "boundary": penalty_boundary,
                "collision": penalty_collision,
                "low_battery": penalty_low_battery,
            },
            "eff_actions": eff_actions,
        }
        return self.observation(), reward, done, info

    # ------------------------------------------------------------------
    # energy ledger
    # ------------------------------------------------------------------
    def _consume_uav_energy(self, joules_per_uav: np.ndarray, kind: str):
        cfg = self.cfg
        joules = np.asarray(joules_per_uav, dtype=float)
        for kk in range(cfg.num_uavs):
            self.uav_soc[kk] = max(self.uav_soc[kk] - float(joules[kk]), 0.0)
        key = {"flight": "flight_energy_j", "comm": "comm_energy_j", "compute": "compute_energy_j"}[kind]
        self.metrics[key] += float(np.sum(joules))
        self.metrics["energy_total_j"] += float(np.sum(joules))

    def consume_comm_energy(self, uav_k: int, joules: float):
        """Public hook for the W2 scheduler to bill communication energy."""
        self._consume_uav_energy(np.eye(self.cfg.num_uavs)[uav_k] * joules, "comm")

    def consume_compute_energy(self, uav_k: int, joules: float):
        """Public hook for the W2 scheduler to bill compute energy."""
        self._consume_uav_energy(np.eye(self.cfg.num_uavs)[uav_k] * joules, "compute")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _gate_actions(self, uav_actions):
        cfg = self.cfg
        if uav_actions is None:
            uav_actions = np.zeros((cfg.num_uavs, 2))
        raw = np.asarray(uav_actions, dtype=float).reshape(cfg.num_uavs, 2)
        norms = np.linalg.norm(raw, axis=1)
        over = norms > cfg.uav_speed_max_mps
        if np.any(over):
            raw[over] = raw[over] * (cfg.uav_speed_max_mps / norms[over, None])
        low_flags = np.zeros(cfg.num_uavs, dtype=bool)
        soc_frac = self.uav_soc / cfg.soc_init_j
        for kk in range(cfg.num_uavs):
            n0 = float(np.linalg.norm(raw[kk]))
            if soc_frac[kk] <= 0.0:
                if n0 > 1e-9:
                    low_flags[kk] = True
                raw[kk] = 0.0
                continue
            if soc_frac[kk] < cfg.soc_low_fraction:
                factor = float(soc_frac[kk] / cfg.soc_low_fraction)  # 0..1, 0 => hover
                if n0 > factor * cfg.uav_speed_max_mps:
                    low_flags[kk] = True
                raw[kk] = raw[kk] * factor
        return raw, low_flags

    def _associate(self):
        """Rule-based association under hard coverage (spec v2 §1.2).

        Rates are masked by the hard-cover radius first; terminals outside
        every UAV's coverage keep ``assignment = -1``.
        """
        cfg = self.cfg
        rates = terminal_uav_rate_matrix(cfg, self.terminals.pos, self.uav_pos)  # (N, K)
        self._rates_nk = hard_cover_rates(cfg, rates, self.terminals.pos, self.uav_pos)
        self.assignment = associate_terminals(self._rates_nk, cfg.num_uavs)

    def _run_scheduling(self):
        """Register pending apps and run ScheduleDAG for this slot (spec §7)."""
        cfg = self.cfg
        # apps are pinned to the association UAV in effect at registration time
        for app in self.apps:
            if app.state != APP_PENDING or app.dag is None:
                continue
            if app.app_id in self.scheduler.apps:
                continue
            u = int(self.assignment[app.terminal])
            if u < 0:
                continue
            self.scheduler.register(
                app.app_id, app.dag, app.terminal, u,
                app.arrival_s, app.deadline_s,
            )

        rates_nk = getattr(self, "_rates_nk", None)
        if rates_nk is None:
            rates_nk = hard_cover_rates(
                cfg,
                terminal_uav_rate_matrix(cfg, self.terminals.pos, self.uav_pos),
                self.terminals.pos,
                self.uav_pos,
            )
        rates_kL = uav_leo_channel_rate(cfg, self.uav_pos, self.leo_sub)
        ctx = SchedCtx(
            t_s=self.t_s,
            t_end=self.t_s + cfg.slot_seconds,
            terminal_pos=self.terminals.pos,
            uav_pos=self.uav_pos,
            leo_sub=self.leo_sub,
            rates_nk=rates_nk,
            rates_kL=rates_kL,
            ephemeris=self.ephemeris,
            decisions=self._build_decision_map(),
        )
        stats = self.scheduler.step(ctx)
        for uav_k, joules in stats.uav_compute:
            self.consume_compute_energy(uav_k, joules)
        for uav_k, joules in stats.uav_comm:
            self.consume_comm_energy(uav_k, joules)
        self.metrics["leo_invalid_attempts"] += stats.leo_rejections
        self.metrics["leo_placements"] += stats.leo_placements
        self.metrics["dep_violations"] += stats.dep_violations
        for aid in stats.completed_apps:
            app = self.apps[aid]  # app ids are list indices
            if app.state != APP_PENDING:
                continue
            finish = self.scheduler.finish_of(aid)
            app.state = APP_DONE
            app.completion_s = finish
            self.metrics["apps_done"] += 1
            self.metrics["done_delay_sum_s"] += max(finish - app.arrival_s, 0.0)

    def _build_decision_map(self):
        """Translate per-UAV per-terminal RL decisions into app_id->device."""
        dec = getattr(self, "_offload_decisions", None)
        if dec is None:
            return None
        cfg = self.cfg
        dec = np.asarray(dec, dtype=int)
        mapping = {}
        cap = int(np.ceil(cfg.num_terminals / cfg.num_uavs))
        for k in range(cfg.num_uavs):
            terms = self._assigned_terminals(k)
            for s, n in enumerate(terms[:cap]):
                d = int(dec[k, s])
                if d == 2:
                    continue
                cand = [
                    a for a in (
                        self.apps[i] for i in self.terminal_apps[n]
                        if self.apps[i].state == APP_PENDING
                    )
                    if a.app_id in self.scheduler.apps
                    and self.scheduler.ready_indices(a.app_id)
                ]
                if not cand:
                    continue
                pick = min(cand, key=lambda a: a.deadline_s)
                mapping[pick.app_id] = int(d)
        return mapping if mapping else None

    def _settle_deadlines(self, t_end: float):
        failed = []
        for i, app in enumerate(self.apps):
            if app.state == APP_PENDING and app.deadline_s <= t_end:
                app.state = APP_FAILED
                failed.append(app)
        return failed

    def current_leo_capacity(self) -> np.ndarray:
        """(K, L) usable UAV-LEO capacity right now (0 outside windows)."""
        visible = np.array(
            [[self.ephemeris.visible(l, self.t_s) for l in range(self.cfg.num_leos)]]
            * self.cfg.num_uavs
        )
        return uav_leo_capacity_matrix(self.cfg, self.uav_pos, self.leo_sub, visible)

    def rate_matrix_terminal_uav(self) -> np.ndarray:
        return terminal_uav_rate_matrix(self.cfg, self.terminals.pos, self.uav_pos)
