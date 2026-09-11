"""ScheduleDAG — embedded polynomial heuristic approximator (spec §7).

Design notes (W2 implementation):

- One-shot slot-granularity scheduling. When a node becomes ready (all
  predecessors done) at or before the current slot boundary, it is *placed*
  once on the cheapest feasible device using current channel/window state.
  Placement reserves absolute-time intervals on transmission assets
  (terminal->UAV uplink, UAV->LEO relay) and on device CPUs (UAV/LEO), so a
  node's work may span several slots while never being re-decided.

- Candidate devices per node: the terminal's pinned associated UAV (apps are
  pinned at arrival so no UAV-UAV transfer is needed) and, when visible, each
  LEO. UAV -> LEO is monotonic: once a node of an app runs on a LEO, later
  nodes of that app may only use that same LEO (no LEO->anything downlink is
  modelled; the sink completes at the device that runs it).

- Visibility-window pruning (the only spec-sanctioned hard rule, §7): a LEO
  placement whose relay/service needs cannot fit inside one visible window is
  forbidden and counted as an invalid LEO attempt. Cross-window LEO work
  therefore never starts under the main method. Spec §1.1: 「跨窗未完成则失败」.
- B6 (vw_prune=False, 常在线/弱窗) only removes this *decision-time* prune:
  the scheduler commits at always-on nominal times and later pays the real
  window cost. Work whose nominal relay+service interval is not covered by a
  contiguous visible window never completes (the app fails at its deadline),
  i.e. ignoring the prune systematically overestimates the LEO path.

- Urgency ordering of ready nodes: 截止期紧迫 > 重节点 > 普通 (then by
  deadline, then heavier first). All placements respect the app deadline:
  a candidate finishing after the deadline is rejected (the node is retried
  in later slots and the app fails only when the deadline actually passes).

- Dependency-violation counter must stay 0 for every legal schedule: nodes are
  only placed once all parents are ``done``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.config import VAConfig
from ..env.ephemeris import Ephemeris
from .dag import DAG

HEAVY_CYCLES_THRESHOLD = 1.0e9  # “重节点”分类阈值（启发式常量）


@dataclass
class SchedCtx:
    """Per-slot scheduling context snapshot computed by the environment."""

    t_s: float                # slot start (absolute seconds)
    t_end: float              # slot end
    terminal_pos: np.ndarray  # (N, 2)
    uav_pos: np.ndarray       # (K, 2)
    leo_sub: np.ndarray       # (L, 2)
    rates_nk: np.ndarray      # (N, K) terminal->UAV
    rates_kL: np.ndarray      # (K, L) UAV->LEO channel (pre-cap)
    ephemeris: Ephemeris
    decisions: Optional[Dict[int, int]] = None  # B4 flat: app_id -> 0 UAV / 1 LEO / 2 wait


@dataclass
class Placement:
    app_id: int
    node_index: int
    device: str                 # "uav" | "leo"
    device_index: int
    start_s: float
    finish_s: float
    comm_j: float = 0.0
    compute_j: float = 0.0
    ul_end: Optional[float] = None       # terminal->UAV uplink completion
    relay_end: Optional[float] = None    # UAV->LEO relay completion
    nominal_finish: Optional[float] = None  # B6 常在线假设下的“名义”完成时刻
    doomed: bool = False                   # B6: always-on commit that can never
                                           # complete (cross-window) -> app fails


@dataclass
class NodeState:
    placed: bool = False
    done: bool = False
    start: Optional[float] = None
    finish: Optional[float] = None
    device: Optional[str] = None
    device_index: Optional[int] = None


@dataclass
class AppState:
    app_id: int
    terminal: int
    uav_k: int                  # pinned association UAV for this app
    dag: DAG
    arrival_s: float
    deadline_s: float
    nodes: List[NodeState] = field(default_factory=list)
    finalized: bool = False
    violated: bool = False  # >=1 node committed before its parents completed

    def __post_init__(self):
        if not self.nodes:
            self.nodes = [NodeState() for _ in self.dag.nodes]


@dataclass
class StepStats:
    placements: int = 0
    dep_violations: int = 0  # child committed before all parents done (only>0 when dag_order=False)
    leo_attempts: int = 0
    leo_rejections: int = 0
    leo_placements: int = 0
    uav_compute: List[Tuple[int, float]] = field(default_factory=list)  # (uav_k, joules)
    uav_comm: List[Tuple[int, float]] = field(default_factory=list)     # (uav_k, joules)
    completed_apps: List[int] = field(default_factory=list)  # app ids done this slot


class ScheduleDAG:
    def __init__(self, cfg: VAConfig):
        self.cfg = cfg
        self.vw_prune = cfg.vw_prune
        self.dag_order = cfg.dag_order
        self.apps: Dict[int, AppState] = {}
        # busy-until times (absolute seconds) per transmission asset / device
        self.busy: Dict[Tuple, float] = {}
        self._windows: List[List[Tuple[float, float]]] = []
        self.dep_violations = 0
        self.leo_attempts = 0
        self.leo_rejections = 0
        self.log: List[Placement] = []   # every committed placement (debug/tests)
        self._leo_rejections_this_step = 0
        self._leo_attempts_this_step = 0

    def bind_ephemeris(self, ephemeris: Ephemeris):
        """Snapshot the deterministic LEO window lists used for pruning."""
        self._windows = [list(ephemeris.windows[l]) for l in range(self.cfg.num_leos)]

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, app_id: int, dag: DAG, terminal: int, uav_k: int,
                 arrival_s: float, deadline_s: float):
        self.apps[app_id] = AppState(
            app_id=app_id, terminal=terminal, uav_k=uav_k, dag=dag,
            arrival_s=arrival_s, deadline_s=deadline_s,
        )

    # ------------------------------------------------------------------
    # per-slot scheduling
    # ------------------------------------------------------------------
    def step(self, ctx: SchedCtx) -> StepStats:
        self._leo_attempts_this_step = 0
        self._leo_rejections_this_step = 0
        self._dep_violations_this_step = 0
        stats = StepStats()
        bound = ctx.t_end

        progressed = True
        while progressed:
            progressed = False
            # 1) mark placed nodes whose reserved finish passed as done
            for app in self.apps.values():
                for ns in app.nodes:
                    if ns.placed and not ns.done and ns.finish is not None:
                        if ns.finish <= bound + 1e-9:
                            ns.done = True
                            progressed = True
            # 2) collect ready nodes
            ready = self._ready_nodes()
            if not ready:
                if not progressed:
                    break
                continue
            ready.sort(key=self._urgency_key(ctx))
            for app, idx in ready:
                placed = self._place_one(app, idx, ctx)
                if placed is not None:
                    stats.placements += 1
                    if placed.device == "uav":
                        stats.uav_compute.append((placed.device_index, placed.compute_j))
                    else:
                        stats.leo_placements += 1
                        if placed.comm_j > 0.0:
                            stats.uav_comm.append((app.uav_k, placed.comm_j))
                    progressed = True
            if not progressed:
                break

        stats.leo_attempts = self._leo_attempts_this_step
        stats.leo_rejections = self._leo_rejections_this_step
        stats.dep_violations = self._dep_violations_this_step
        stats.completed_apps = self._finalize_completed(bound)
        return stats

    def _ready_nodes(self):
        ready = []
        for app in self.apps.values():
            if app.finalized:
                continue
            for idx, ns in enumerate(app.nodes):
                if ns.placed or ns.done:
                    continue
                parents = app.dag.nodes[idx].parents
                if self.dag_order and not all(app.nodes[p].done for p in parents):
                    continue
                if not self.dag_order or all(app.nodes[p].done for p in parents):
                    ready.append((app, idx))
        return ready

    def _urgency_key(self, ctx: SchedCtx):
        def key(item):
            app, idx = item
            remaining = app.deadline_s - ctx.t_s
            if remaining <= self.cfg.deadline_margin_s:
                cls = 0  # 截止期紧迫
            elif app.dag.nodes[idx].cycles > HEAVY_CYCLES_THRESHOLD:
                cls = 1  # 重节点
            else:
                cls = 2  # 普通
            return (cls, app.deadline_s, -app.dag.nodes[idx].cycles)
        return key

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------
    def _place_one(self, app: AppState, idx: int, ctx: SchedCtx) -> Optional[Placement]:
        node = app.dag.nodes[idx]
        parents_done = [app.nodes[p] for p in node.parents]
        parent_leo = [ns.device_index for ns in parents_done if ns.device == "leo"]

        decision = None
        if ctx.decisions is not None:
            decision = ctx.decisions.get(app.app_id, 2)  # B4: default wait

        candidates: List[Tuple[str, int]] = []
        if decision is not None:
            if decision == 0:
                if parent_leo:
                    return None  # data on LEO cannot move back to a UAV
                candidates = [("uav", app.uav_k)]
            elif decision == 1:
                candidates = (
                    [("leo", parent_leo[0])] if parent_leo
                    else [("leo", l) for l in range(self.cfg.num_leos)]
                )
            else:  # 2 = wait
                return None
        else:
            if not parent_leo:
                candidates.append(("uav", app.uav_k))
            if parent_leo:
                candidates.append(("leo", parent_leo[0]))  # stay on that LEO
            else:
                for l in range(self.cfg.num_leos):
                    candidates.append(("leo", l))

        best: Optional[Placement] = None
        for device, dev_idx in candidates:
            p = self._try_place(app, idx, ctx, device, dev_idx)
            if device == "leo":
                self.leo_attempts += 1
                self._leo_attempts_this_step += 1
                if p is None or p.doomed:
                    # rejected now, or committed but can never complete inside
                    # a window (B6): both are invalid LEO attempts
                    self.leo_rejections += 1
                    self._leo_rejections_this_step += 1
                if p is None:
                    continue
            elif p is None:
                continue  # UAV rejected (deadline infeasible): retry next slot
            if best is None or p.finish_s < best.finish_s - 1e-9 or (
                abs(p.finish_s - best.finish_s) <= 1e-9 and p.device == "uav"
            ):
                best = p

        if best is None:
            return None
        self._commit(app, idx, best)
        return best

    def ready_indices(self, app_id: int) -> List[int]:
        """Indices of ready-but-unplaced nodes of an app (exposed to RL obs)."""
        app = self.apps[app_id]
        out = []
        for idx, ns in enumerate(app.nodes):
            if ns.placed or ns.done:
                continue
            parents = app.dag.nodes[idx].parents
            if self.dag_order and not all(app.nodes[p].done for p in parents):
                continue
            if not self.dag_order or all(app.nodes[p].done for p in parents):
                out.append(idx)
        return out

    def _try_place(self, app: AppState, idx: int, ctx: SchedCtx,
                   device: str, dev_idx: int) -> Optional[Placement]:
        """Evaluate one candidate without mutating state."""
        node = app.dag.nodes[idx]
        if device == "uav":
            k = dev_idx
            ul_end = None
            ready = max(ctx.t_s, app.arrival_s)
            if not node.parents or not self.dag_order:
                # source input must first travel terminal n -> UAV k; in the
                # no-DAG-order ablation every node is an independent package
                # whose input still starts at its own terminal
                bits = float(node.in_bits)
                r_nk = float(ctx.rates_nk[app.terminal, k])
                ul_start = max(
                    ready, self._busy(("ul", app.terminal))
                )
                ul_end = ul_start + bits / max(r_nk, 1.0)
                ready = max(ready, ul_end)
            else:
                for p in node.parents:
                    ready = max(ready, float(app.nodes[p].finish))
            start = max(ready, self._busy(("uav", k)))
            dur = node.cycles / self.cfg.uav_cpu_cycles_per_s
            finish = start + dur
            if finish > app.deadline_s + 1e-9:
                return None
            return Placement(
                app_id=app.app_id, node_index=idx, device="uav",
                device_index=k, start_s=start, finish_s=finish,
                comm_j=0.0, compute_j=self.cfg.uav_power_watt * dur,
                ul_end=ul_end,
            )

        # ---- LEO candidate --------------------------------------------
        l = dev_idx
        k = app.uav_k
        r_kL = float(ctx.rates_kL[k, l])
        r_nk = float(ctx.rates_nk[app.terminal, k])
        if r_kL <= 0.0:
            return None
        ul_end = None
        data_ready = max(ctx.t_s, app.arrival_s)
        relay_bits = 0.0
        comm_j = 0.0

        if not node.parents or not self.dag_order:
            # source input (or, under the no-DAG-order ablation, every node's
            # own input): terminal n -> UAV k (uplink) then relay to LEO
            bits = float(node.in_bits)
            relay_bits += bits
            ul_start = max(data_ready, self._busy(("ul", app.terminal)))
            ul_end = ul_start + bits / max(r_nk, 1.0)
            data_ready = max(data_ready, ul_end)
        else:
            for p in node.parents:
                ps = app.nodes[p]
                if ps.device == "leo" and ps.device_index == l:
                    data_ready = max(data_ready, float(ps.finish))
                else:
                    # bits produced by parent p must be relayed UAV k -> LEO l
                    bits = float(app.dag.nodes[p].out_bits)
                    relay_bits += bits
                    data_ready = max(data_ready, float(ps.finish))
        relay_cost_s = relay_bits / max(r_kL, 1.0)
        comm_j = self.cfg.comm_power_watt * relay_cost_s

        dur = node.cycles / self.cfg.leo_cpu_cycles_per_s
        relay_asset = ("relay", k, l)
        if self.vw_prune:
            # find a single visibility window that fits [relay start, service end]
            res = self._fit_leo_window(l, k, data_ready, relay_cost_s, dur,
                                       app.deadline_s)
            if res is None:
                return None
            start_s, finish_s, relay_end = res
            return Placement(
                app_id=app.app_id, node_index=idx, device="leo", device_index=l,
                start_s=start_s, finish_s=finish_s, comm_j=comm_j, compute_j=0.0,
                ul_end=ul_end, relay_end=relay_end,
            )
        # B6 (常在线/弱窗决策): decide without the visibility prune. Reality is
        # not always-on: relay+service must sit inside one contiguous visible
        # window. If the always-on nominal interval crosses an outage the work
        # is doomed ("跨窗未完成则失败") and the app fails at its deadline.
        relay_start = max(data_ready, self._busy(relay_asset))
        relay_end = relay_start + relay_cost_s
        start_s = max(relay_end, self._busy(("leo", l)))
        nominal_finish = start_s + dur
        if nominal_finish > app.deadline_s + 1e-9:
            return None
        covered = any(
            t_in - 1e-9 <= relay_start and nominal_finish <= t_out + 1e-9
            for t_in, t_out in self._windows[l]
        )
        if not covered:
            # doomed commit: relayed but never served, reserved past the
            # horizon so it can never become "done" inside the episode; the
            # pending app fails at its deadline. The LEO CPU is NOT reserved
            # past the failed window so later work is not jammed.
            return Placement(
                app_id=app.app_id, node_index=idx, device="leo", device_index=l,
                start_s=start_s, finish_s=app.deadline_s + self.cfg.horizon
                * self.cfg.slot_seconds + 1.0,
                comm_j=comm_j, compute_j=0.0, ul_end=ul_end,
                relay_end=relay_end, doomed=True,
            )
        return Placement(
            app_id=app.app_id, node_index=idx, device="leo", device_index=l,
            start_s=start_s, finish_s=nominal_finish, comm_j=comm_j,
            compute_j=0.0, ul_end=ul_end, relay_end=relay_end,
            nominal_finish=nominal_finish,
        )

    def _fit_leo_window(self, l: int, k: int, data_ready: float,
                        relay_cost_s: float, dur: float,
                        deadline_s: float) -> Optional[Tuple[float, float, float]]:
        """Return (service_start, finish) fitting one LEO visibility window."""
        relay_asset = ("relay", k, l)
        relay_free = self._busy(relay_asset)
        leo_free = self._busy(("leo", l))
        windows = self._ephemeris_windows(l)
        for t_in, t_out in windows:
            if t_out <= data_ready + 1e-9:
                continue
            # relay can only run inside the window
            relay_start = max(data_ready, relay_free, t_in)
            relay_end = relay_start + relay_cost_s
            if relay_end > t_out:
                continue
            service_start = max(relay_end, leo_free, t_in)
            finish = service_start + dur
            if finish > t_out + 1e-9:
                continue
            if finish > deadline_s + 1e-9:
                return None
            return service_start, finish, relay_end
        return None

    def _ephemeris_windows(self, l: int):
        # ScheduleDAG keeps its own window list copy so it never depends on a
        # live env; set by bind_ephemeris().
        return self._windows[l]

    # ------------------------------------------------------------------
    # commit / bookkeeping
    # ------------------------------------------------------------------
    def _commit(self, app: AppState, idx: int, p: Placement):
        node = app.dag.nodes[idx]
        if node.parents and not all(app.nodes[pidx].done for pidx in node.parents):
            # dag_order=False treats dependent nodes as independent packages: a
            # child is committed before its parents produced their outputs. A
            # legal DAG-order schedule never triggers this (guarded by _ready_nodes).
            self.dep_violations += 1
            self._dep_violations_this_step += 1
            app.violated = True
        ns = app.nodes[idx]
        ns.placed = True
        ns.start = p.start_s
        ns.finish = p.finish_s
        ns.device = p.device
        ns.device_index = p.device_index
        if p.device == "uav":
            self._set_busy(("uav", p.device_index), p.finish_s)
        elif p.device == "leo":
            if not p.doomed:
                self._set_busy(
                    ("leo", p.device_index),
                    p.nominal_finish if p.nominal_finish is not None
                    else p.finish_s,
                )
        if p.ul_end is not None:
            self._set_busy(("ul", app.terminal), p.ul_end)
        if p.relay_end is not None:
            self._set_busy(("relay", app.uav_k, p.device_index), p.relay_end)
        self.log.append(p)

    def _finalize_completed(self, bound: float) -> List[int]:
        done = []
        for app in self.apps.values():
            if app.finalized:
                continue
            sink_ns = app.nodes[app.dag.sink]
            if sink_ns.done and sink_ns.finish is not None:
                if sink_ns.finish <= bound + 1e-9:
                    app.finalized = True
                    done.append(app.app_id)
        return done

    def finish_of(self, app_id: int) -> float:
        return float(self.apps[app_id].nodes[self.apps[app_id].dag.sink].finish)

    def _busy(self, key: Tuple) -> float:
        return float(self.busy.get(key, 0.0))

    def _set_busy(self, key: Tuple, value: float):
        self.busy[key] = max(self._busy(key), value)
