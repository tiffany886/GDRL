"""
utils/logger.py — GDRL 项目统一日志工具
========================================
使用方式：
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("开始训练，场景：U=%d L=%d N=%d", args.U, args.L, args.N)

日志同时输出到：
  - 控制台（彩色，INFO 级别）
  - 文件 logs/gdrl_YYYYMMDD_HHMMSS.log（DEBUG 级别，保留完整记录）

注意：第一次调用 get_logger() 时会自动创建 logs/ 目录。
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── 日志目录（相对项目根目录）────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"

# ── 是否已完成根 logger 的初始化（防止重复添加 handler）───────
_INITIALIZED = False


def _setup_root_logger(log_dir: Path = _LOG_DIR) -> None:
    """
    初始化根 logger：
      - 控制台 handler：INFO 级别，带颜色（Windows 兼容）
      - 文件 handler：DEBUG 级别，写入带时间戳的日志文件
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger("gdrl")
    root.setLevel(logging.DEBUG)   # 根 logger 接受所有级别，由各 handler 过滤

    # ── 日志格式 ────────────────────────────────────────────
    fmt_console = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    fmt_file    = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    datefmt     = "%Y-%m-%d %H:%M:%S"

    # ── 控制台 handler ──────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt_console, datefmt=datefmt))

    # ── 文件 handler ────────────────────────────────────────
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gdrl_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt_file, datefmt=datefmt))

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    root.info("日志系统初始化完成，日志文件：%s", log_file)
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    获取命名 logger，首次调用时自动完成根 logger 初始化。

    参数：
        name: 通常传入 __name__，便于追踪日志来源文件

    返回：
        logging.Logger 实例

    示例：
        logger = get_logger(__name__)
        logger.debug("调试信息：obs shape = %s", obs.shape)
        logger.info("Episode %d 完成，reward = %.2f", ep, reward)
        logger.warning("信道模型使用近似版本，结果与论文可能有偏差")
        logger.error("AMN 权重加载失败：%s", exc)
    """
    _setup_root_logger()
    # 统一使用 "gdrl." 前缀，方便全局过滤
    return logging.getLogger(f"gdrl.{name}" if not name.startswith("gdrl") else name)
