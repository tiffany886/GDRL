import json
from pathlib import Path


# 三组实验场景。受当前动作编码限制，LEO/HAPS 的编号最多支持 0-15，
# 因此 L 和 N 暂时不要超过 16；若以后要做更大网络，需要扩展动作编码位数。
SCENARIOS = {
    "small": {
        "U": 3,
        "L": 8,
        "N": 8,
        "T": 100,
        "description": "小规模场景：用于快速验证代码流程和基础对比。",
    },
    "medium": {
        "U": 6,
        "L": 12,
        "N": 12,
        "T": 100,
        "description": "中规模场景：用户和资源节点增加，开始体现图结构建模价值。",
    },
    "large": {
        "U": 10,
        "L": 16,
        "N": 16,
        "T": 100,
        "description": "大规模场景：当前编码范围内较复杂的 SAGIN 拓扑。",
    },
}


def _is_hybrid(args):
    """检测是否为 hybrid 四层异构场景（包含 gNB/UAV/MEC 节点）。"""
    return getattr(args, 'G', 0) > 0 or getattr(args, 'V', 0) > 0 or getattr(args, 'M', 0) > 0


def add_scenario_arguments(parser):
    parser.add_argument(
        "--scenario",
        choices=["custom", *SCENARIOS.keys()],
        default="custom",
        help="选择预设场景：small / medium / large；custom 表示使用手动 U/L/N/T。",
    )
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="experiments",
        help="统一实验输出根目录，默认保存到 ./experiments。",
    )
    return parser


def apply_scenario(args):
    """把 small/medium/large 的参数写回 args，兼容原来的手动 U/L/N/T 用法。"""
    scenario = getattr(args, "scenario", "custom")
    if scenario != "custom":
        cfg = SCENARIOS[scenario]
        args.U = cfg["U"]
        args.L = cfg["L"]
        args.N = cfg["N"]
        args.T = cfg["T"]
    return args


def validate_scenario(args):
    """检查当前场景是否满足动作编码和邻接矩阵生成逻辑的基本约束。"""
    hybrid = _is_hybrid(args)
    if not hybrid:
        if args.L > 16 or args.N > 16:
            raise ValueError(
                "当前离散动作只用 4 bit 表示 LEO/HAPS 编号，所以 L 和 N 暂时都不能超过 16。"
            )
        if args.L < args.U + 4 or args.N < args.U + 4:
            raise ValueError(
                "当前拓扑生成器要求 L >= U + 4 且 N >= U + 4，"
                "这样每个用户才能连接直接节点和若干间接节点。"
            )


def scenario_tag(args):
    scenario = getattr(args, "scenario", "custom")
    hybrid = _is_hybrid(args)
    if hybrid:
        G = getattr(args, 'G', 0); V = getattr(args, 'V', 0); M = getattr(args, 'M', 0)
        return f"{scenario}_U{args.U}_G{G}_V{V}_L{args.L}_M{M}_T{args.T}"
    return f"{scenario}_U{args.U}_L{args.L}_N{args.N}_T{args.T}"


def scenario_root(args):
    return Path(args.experiment_root) / scenario_tag(args)


def default_compare_dir(args):
    return scenario_root(args) / f"compare_{args.episodes}ep_T{args.T}"


def default_single_dir(args):
    return scenario_root(args) / f"single_{args.total_step}steps_T{args.T}"


def amn_paths(args):
    tag = scenario_tag(args)
    amn_dir = scenario_root(args) / "amn"
    return {
        "dir": amn_dir,
        "dis": amn_dir / f"autoencoder_dis_{tag}.pth",
        "con": amn_dir / f"autoencoder_con_{tag}.pth",
        "loss": amn_dir / f"amn_loss_{tag}.csv",
    }


def write_manifest(path, args, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scenario = getattr(args, "scenario", "custom")
    hybrid = _is_hybrid(args)
    data = {
        "scenario": scenario,
        "scenario_tag": scenario_tag(args),
        "U": args.U,
        "L": args.L,
        "N": args.N,
        "T": args.T,
        "description": SCENARIOS.get(scenario, {}).get("description", "自定义场景"),
    }
    if hybrid:
        data["G"] = getattr(args, 'G', 0)
        data["V"] = getattr(args, 'V', 0)
        data["M"] = getattr(args, 'M', 0)
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
