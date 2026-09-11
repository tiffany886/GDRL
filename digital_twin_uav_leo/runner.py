"""Twin-driven episode evaluation: feed twin obs to a policy, execute in the
physical env, and report reward / latency / energy / task-level completion /
offload-decision agreement / sync count.
"""
import numpy as np

from .dt_env import TwinnedWorld
from .adaptive_sync import local_ingress_error, change_point_error
from uav_leo_experiment.multi_uav import exact_offload_multi


def run_dt_episode(env, policy, cfg, tau=3, predictor="linear", eps=0.0,
                   sync_mode="fixed", seed=1, probe_m=2, probe_n=3, delta=10.0,
                   task_mode="per_user", sync_delay=0, sync_loss=0.0):
    rng = np.random.default_rng(0)
    obs = env.reset(seed=seed)
    ingress_ewma = None
    probe_rounds = 0      # actual probe *transmissions* (extra signalling)
    eval_rounds = 0       # trigger evaluations (no signalling)
    trace = []
    tw = TwinnedWorld(env, tau=tau, predictor=predictor, eps=eps, rng=rng,
                      sync_mode=sync_mode, probe_m=probe_m, probe_n=probe_n,
                      delta=delta, task_mode=task_mode,
                      sync_delay=sync_delay, sync_loss=sync_loss,
                      chan_rng=np.random.default_rng(9973 + int(seed)))
    tw.sync(obs, immediate=True)   # initial snapshot is always available
    ep_reward = 0.0
    ep_latency = 0.0
    ep_energy = 0.0
    n_task = 0
    n_ok = 0
    n_agree = 0
    n_dec = 0
    done = False
    K = int(cfg.uavs)
    while not done:
        prev_mask = np.asarray(obs["task_bits"], dtype=float) > 1e-6
        prev_pos = np.asarray(obs["uav_pos"], dtype=float)
        obs_dt = tw.get_obs()
        action = policy.act(obs_dt, rng, cfg)
        move = np.asarray(action["move"], dtype=float)
        if move.ndim == 1:
            move = np.tile(move, (K, 1))
        pos_after = np.clip(prev_pos + move * cfg.slot_seconds, 0.0, cfg.area_size)
        perfect_t, _ = exact_offload_multi(dict(obs), cfg, pos_after)
        cur_task_bits = np.asarray(obs["task_bits"], dtype=float)
        obs, reward, done, info = env.step(action)
        n_dec += int(cfg.users)
        n_agree += int(np.sum(np.asarray(perfect_t) == np.asarray(action["targets"])))
        deadline = cfg.success_deadline_s
        lat_arr = np.asarray(info["per_user_latency"], dtype=float)
        n_task += int(prev_mask.sum())
        n_ok += int(np.sum(prev_mask & (lat_arr <= deadline)))
        ep_reward += reward
        ep_latency += info["latency_mean"]
        ep_energy += info["total_energy"]
        tw.step_forward(dt=float(cfg.slot_seconds))
        if sync_mode == "adaptive":
            if task_mode == "local_arr":
                tgt = np.asarray(info["target_names"])
                recv = cur_task_bits[tgt == "uav"]
                tgt_tw = np.asarray(action["targets"], dtype=int)
                K = int(cfg.uavs)
                served = (tgt_tw >= 1) & (tgt_tw <= K)
                exp = np.asarray(obs_dt["task_bits"], dtype=float)[served]
                eval_rounds += 1   # purely local, zero signalling
                err = local_ingress_error(
                    recv, exp, float(getattr(cfg, "task_bits_min", 3.0e5)))
                fired = err > delta
                if fired:
                    tw.sync(obs)
                trace.append((int(tw._t), float(recv.sum()), float(exp.sum()),
                              bool(fired)))
            elif task_mode == "local_cp":
                tgt = np.asarray(info["target_names"])
                recv = cur_task_bits[tgt == "uav"]
                ref = float(getattr(cfg, "task_bits_min", 3.0e5))
                if ingress_ewma is None:
                    ingress_ewma = float(recv.sum()) if recv.size else ref
                    err, cur = 0.0, ingress_ewma
                else:
                    err, cur = change_point_error(ingress_ewma, recv, ref)
                eval_rounds += 1   # purely local, zero signalling
                alpha = 0.3
                ingress_ewma = (1.0 - alpha) * float(ingress_ewma) + alpha * float(cur)
                fired = err > delta
                if fired:
                    tw.sync(obs)
                trace.append((int(tw._t), float(recv.sum()), float(ingress_ewma),
                              bool(fired)))
            else:
                tw._maybe_probe_and_resync(obs)
                if tw._probe_counter == 0:
                    probe_rounds += 1
        elif tw.since_sync >= tau and not tw.has_pending:
            tw.sync(obs)
    slots = max(int(cfg.horizon), 1)
    return {
        "reward": ep_reward,
        "latency": ep_latency / slots,
        "energy": ep_energy / slots,
        "completion": n_ok / max(n_task, 1),
        "agree": n_agree / max(n_dec, 1),
        "syncs": tw.sync_count,
        "probe_rounds": probe_rounds,
        "eval_rounds": eval_rounds,
        "dropped_syncs": tw.dropped_syncs,
        "delivered_syncs": tw.delivered_syncs,
        "trace": trace,
    }
