"""
setup.py — 以开发模式安装 GDRL 包
用法：pip install -e .
安装后可在任意目录使用：from gdrl.envs import NetworkEnvironment
"""
from setuptools import setup, find_packages

setup(
    name="gdrl",
    version="0.1.0",
    description="Graph Deep Reinforcement Learning for SAGIN Resource Allocation (IEEE JSAC 2025)",
    packages=find_packages(include=["gdrl", "gdrl.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "torch_geometric>=2.4.0",
        "stable-baselines3>=2.2.1",
        "sb3-contrib>=2.2.1",
        "gymnasium>=0.29.1",
        "numpy>=1.23.5",
        "scipy>=1.10.1",
        "PyYAML>=6.0",
    ],
)
