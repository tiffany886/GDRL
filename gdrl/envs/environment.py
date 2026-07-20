"""
gdrl/envs/environment.py — SAGIN 网络强化学习环境
==================================================
NetworkEnvironment 实现 gymnasium.Env 接口，每个 step 包含：
  1. 通过 AMN（encoder_dis + encoder_con）解码连续潜在动作
  2. 根据离散卸载决策和连续资源分配更新节点资源
  3. 计算通信速率、处理时延和奖励
  4. 更新状态并返回下一观测
"""
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from gdrl.envs.channel import ChannelModel
from gdrl.envs.rate import rate_calculation
from gdrl.core.energy import tx_energy, comp_energy, local_energy
from gdrl.core.env_init import ResetFunction


def _row_tensor(value):
    if torch.is_tensor(value):
        return value.detach().clone().reshape(1)
    return torch.tensor(value).reshape(1)


class NetworkEnvironment(gym.Env):
    MIN_COMPUTE_SHARE = 0.05
    MIN_CHANNEL_SHARE = 1e-3
    MIN_RATE = 1.0
    MAX_LATENCY_SECONDS = 1.0

    def __init__(self, U, L, N, T, user_requests, user_lists, save_var, load_var, encoder_dis, encoder_con):
        super(NetworkEnvironment, self).__init__()
        # 动作空间：TRPO 输出的 32 维连续向量（[-10, 10]），经 AMN 映射为结构化动作
        lower_bounds = np.ones(32) * -10
        upper_bounds = np.ones(32) * 10
        self.action_space = spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float32)

        # 观测空间：8U+3L+3N+2 维连续向量
        lower_bounds_state = np.ones(8*U+3*L+3*N+2)*-np.inf
        upper_bounds_state = np.ones(8*U+3*L+3*N+2)*np.inf
        self.observation_space = spaces.Box(low=lower_bounds_state, high=upper_bounds_state, dtype=np.float32)

        self.U = U; self.L = L; self.N = N; self.T = T
        self.gamma1 = 0.8; self.gamma2 = 1; self.c = 3e8
        try:
            from arg_parser import get_args as _get_args
            self.gamma3 = getattr(_get_args(), 'energy_weight', 0.1)
        except Exception:
            self.gamma3 = 0.1
        self.user_requests = user_requests
        self.save_var = save_var; self.load_var = load_var
        self.user_lists = user_lists
        self.encoder_dis = encoder_dis; self.encoder_con = encoder_con

    def step(self, action):
        tua_max = 10e9
        LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter = self.load_var()

        Uf = self.user_requests['Uf'][:, counter]
        Pu = self.user_requests['Pu'][:, counter]
        Su = self.user_requests['Su'][:, counter]
        Ou = self.user_requests['Ou'][:, counter]
        Vu = self.user_requests['Vu'][:, counter]
        Lu = self.user_requests['Lu'][:, counter]
        C_l = LEO_status['C_l']; LEO_place = LEO_status['LEO_place']; C_l_ori = LEO_status['C_l_ori']
        C_n = HAPS_status['C_n']; HAPS_place = HAPS_status['HAPS_place']; C_n_ori = HAPS_status['C_n_ori']
        U_place = user_status['U_place']; C_u_ori = user_status['C_u_ori']
        C_resource_l = LEO_resource_status['C_resource_l']
        C_resource_n = HAPS_resource_status['C_resource_n']

        # AMN：将 TRPO 的 32 维连续动作拆为离散（前 16 维）+ 连续（后 16 维）
        action = torch.tensor(action).type(torch.float32)
        dis_action = self.encoder_dis(action[0:16].unsqueeze(0), self.user_lists).squeeze()
        con_action = self.encoder_con(action[16:32].unsqueeze(0))
        con_action = torch.nan_to_num(con_action, nan=0.0, posinf=1.0, neginf=0.0)
        compute_share = torch.clamp(con_action[:, 0], min=self.MIN_COMPUTE_SHARE, max=1.0)
        channel_share = torch.clamp(con_action[:, 1], min=self.MIN_CHANNEL_SHARE)
        channel_share = channel_share / torch.clamp(channel_share.sum(), min=self.MIN_CHANNEL_SHARE)
        con_action = torch.stack([compute_share, channel_share], dim=1)

        # 解析离散动作（7-bit 二进制字符串 → 逐位 tensor）
        dis_action = [format(num, '07b') for num in dis_action]
        s = torch.tensor([[int(c) for c in d] for d in dis_action])
        part_length = len(s) // self.U
        parts = [s[i:i + part_length] for i in range(0, len(s), part_length)]

        C_l_new = C_l; C_n_new = C_n; A_u_new = torch.as_tensor(A_u, dtype=torch.float32)
        reward = 0; latency_sum = torch.tensor(0.0); energy_sum = 0.0; u = 0

        for part in parts:
            part = part.reshape(-1).tolist()
            index = int(''.join(str(d) for d in part[-4:]), 2)
            if part[2] == 1:
                _c_u = float(C_u_ori[u]) if u < len(C_u_ori) else 1e9
                local_proc_time = float(Su[u] * Ou[u]) / max(_c_u, 1.0)
                _local_e = local_energy(local_proc_time)
                energy_sum += _local_e
                reward += -self.gamma3 * _local_e * 1e3
            else:
                if part[0] == 0:   # LEO 卸载
                    if C_l_new[index] == 0:
                        C_l_new[index] += C_resource_l[index, counter]; reward -= 5; break
                    if A_u_new == 0:
                        A_u_new += A_resource[counter]; reward -= 5; break
                    process_time = (Su[u] * Ou[u]) / (con_action[u, 0] * C_l[index] * tua_max)
                    number_mini_slot = torch.ceil(process_time * 1e4)
                    index_time = int(min(counter+number_mini_slot, C_resource_l.shape[1]-1))
                    C_resource_l[index, index_time] += torch.ceil(con_action[u, 0] * C_l[index])
                    A_resource[int(min(counter+number_mini_slot, A_resource.shape[0]-1))] += torch.ceil(con_action[u, 1] * A_u)
                    C_l_new[index] = min(C_l_new[index] - torch.ceil(con_action[u, 0] * C_l[index]) + C_resource_l[index, counter], C_l_ori[index])
                    A_u_new = min(A_u_new - torch.ceil(con_action[u, 1] * A_u) + A_resource[counter], A_u_ori)
                    C_l_new[index] = max(C_l_new[index], 0); A_u_new = max(A_u_new, 0)
                    hu = ChannelModel(Uf[u], LEO_place[index])
                else:              # HAPS 卸载
                    if C_n_new[index] == 0:
                        C_n_new[index] = C_n_ori[index]; reward -= 5; break
                    if A_u_new == 0:
                        A_u_new += A_resource[counter]; reward -= 5; break
                    process_time = (Su[u] * Ou[u]) / (con_action[u, 0] * C_n[index] * tua_max)
                    number_mini_slot = torch.ceil(process_time * 1e4)
                    C_resource_n[index, int(min(counter+number_mini_slot, C_resource_n.shape[1]-1))] += torch.ceil(con_action[u, 0] * C_n[index])
                    A_resource[int(min(counter+number_mini_slot, A_resource.shape[0]-1))] += torch.ceil(con_action[u, 1] * A_u)
                    C_n_new[index] = min(C_n_new[index] - torch.ceil(con_action[u, 0] * C_n[index]) + C_resource_n[index, counter], C_n_ori[index])
                    A_u_new = min(A_u_new - torch.ceil(con_action[u, 1] * A_u) + A_resource[counter], A_u_ori)
                    C_n_new[index] = max(C_n_new[index], 0); A_u_new = max(A_u_new, 0)
                    hu = ChannelModel(Uf[u], HAPS_place[index])

                base_rate = float(np.asarray(rate_calculation(Pu[u], hu, Uf[u])).reshape(-1)[0])
                if not np.isfinite(base_rate): base_rate = self.MIN_RATE
                base_rate = max(base_rate, self.MIN_RATE)
                Ru_k = torch.clamp(con_action[u, 1] * float(A_u) * base_rate, min=self.MIN_RATE)

                if part[1] == 0:
                    latency = process_time + Su[u] / Ru_k
                else:
                    if part[0] == 0:
                        d = min(np.abs(LEO_place[index] - LEO_place[u]), np.abs(LEO_place[index] - LEO_place[u+1]))
                    else:
                        d = min(np.abs(HAPS_place[index] - HAPS_place[u]), np.abs(HAPS_place[index] - HAPS_place[u+1]))
                    latency = process_time + Su[u] / Ru_k + d / self.c

                latency = torch.nan_to_num(torch.as_tensor(latency, dtype=torch.float32),
                                           nan=self.MAX_LATENCY_SECONDS,
                                           posinf=self.MAX_LATENCY_SECONDS,
                                           neginf=self.MAX_LATENCY_SECONDS)
                latency = torch.clamp(latency, min=0.0, max=self.MAX_LATENCY_SECONDS)
                _tx_e = tx_energy(
                    float(Pu[u]),
                    float(Su[u]),
                    float(Ru_k.detach()) if hasattr(Ru_k, 'detach') else float(Ru_k)
                )
                _proc_time = float(process_time.detach()) if hasattr(process_time, 'detach') else float(process_time)
                _node_type = "leo" if part[0] == 0 else "haps"
                _alloc_vms = float(torch.ceil(con_action[u, 0] * (C_l[index] if part[0] == 0 else C_n[index])))
                _comp_e = comp_energy(_node_type, _alloc_vms, _proc_time)
                step_energy = _tx_e + _comp_e
                energy_sum += step_energy
                reward += (self.gamma1 * (Lu[u] - latency * 1e4)
                           - self.gamma2 * latency * 1e3
                           - self.gamma3 * step_energy * 1e3)
                latency_sum += latency
            u += 1

        LEO_status = {'C_l': C_l_new, 'LEO_place': LEO_place, 'C_l_ori': C_l_ori}
        HAPS_status = {'C_n': C_n_new, 'HAPS_place': HAPS_place, 'C_n_ori': C_n_ori}
        user_status = {'U_place': U_place, 'C_u_ori': C_u_ori}
        LEO_resource_status = {'C_resource_l': C_resource_l}
        HAPS_resource_status = {'C_resource_n': C_resource_n}

        A_u_new = _row_tensor(A_u_new); A_u_ori = _row_tensor(A_u_ori)
        state = np.concatenate([Uf, Pu, Su, Ou, Vu, Lu, C_l_new, C_n_new, A_u_new, A_u_ori,
                                 C_u_ori, C_l_ori, C_n_ori, U_place, LEO_place, HAPS_place]).astype(np.float32)
        counter += 1
        A_u_new = A_u_new.squeeze(0); A_u_ori = A_u_ori.squeeze(0)
        self.save_var(LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
                      user_status, A_u_new, A_u_ori, A_resource, counter)
        done = (counter == self.T)
        reward = float(reward)
        if not np.isfinite(reward): reward = -float(self.U) * 1e4
        info = {'latency': latency_sum.detach().clone(), 'energy': energy_sum,
                'dis_action': dis_action, 'con_action': con_action.detach().clone()}
        return state, reward, done, False, info

    def reset(self, seed=None, options=None):
        counter = 0
        LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource = ResetFunction(self.N, self.L, self.T, self.U)
        self.save_var(LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
                      user_status, A_u, A_u_ori, A_resource, counter)
        Uf = self.user_requests['Uf'][:, counter]; Pu = self.user_requests['Pu'][:, counter]
        Su = self.user_requests['Su'][:, counter]; Ou = self.user_requests['Ou'][:, counter]
        Vu = self.user_requests['Vu'][:, counter]; Lu = self.user_requests['Lu'][:, counter]
        C_l = LEO_status['C_l']; C_l_ori = LEO_status['C_l_ori']; LEO_place = LEO_status['LEO_place']
        C_n = HAPS_status['C_n']; C_n_ori = HAPS_status['C_n_ori']; HAPS_place = HAPS_status['HAPS_place']
        U_place = user_status['U_place']; C_u_ori = user_status['C_u_ori']
        A_u = _row_tensor(A_u); A_u_ori = _row_tensor(A_u_ori)
        state = np.concatenate([Uf, Pu, Su, Ou, Vu, Lu, C_l, C_n, A_u, A_u_ori,
                                 C_u_ori, C_l_ori, C_n_ori, U_place, LEO_place, HAPS_place]).astype(np.float32)
        return state, {}
