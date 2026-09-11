"""
gdrl/core/update.py — 环境状态变量的 save/load 闭包
====================================================
updatevalue() 返回两个闭包：
  save_variable(...)  : 将新状态存入闭包作用域
  load_variable()     : 从闭包读取当前状态
这种设计允许 NetworkEnvironment 在多步 step() 中共享可变状态。
"""
from docs.GDRL.gdrl.core.env_init import ResetFunction
from docs.GDRL.arg_parser import get_args


def updatevalue():
    """
    创建 SAGIN 环境的状态保存/加载闭包对。

    返回：
        (save_variable, load_variable) — 通过闭包共享同一套状态变量
    """
    args = get_args()
    counter = 0
    LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
        user_status, A_u, A_u_ori, A_resource = ResetFunction(args.N, args.L, args.T, args.U)

    def save_variable(new_LEO_status, new_HAPS_status, new_LEO_resource_status,
                      new_HAPS_resource_status, new_user_status, new_A_u,
                      new_A_u_ori, new_A_resource, new_counter):
        nonlocal LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter
        LEO_status = new_LEO_status
        HAPS_status = new_HAPS_status
        LEO_resource_status = new_LEO_resource_status
        HAPS_resource_status = new_HAPS_resource_status
        user_status = new_user_status
        A_u = new_A_u
        A_u_ori = new_A_u_ori
        A_resource = new_A_resource
        counter = new_counter
        return (LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
                user_status, A_u, A_u_ori, A_resource, counter)

    def load_variable():
        nonlocal LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter
        return (LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
                user_status, A_u, A_u_ori, A_resource, counter)

    return save_variable, load_variable
