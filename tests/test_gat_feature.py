import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_use_gat_flag_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--U", "3", "--L", "8", "--N", "8"])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    args = arg_parser.get_args()
    assert args.use_gat is True


def test_use_gat_flag_disabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--U", "3", "--L", "8", "--N", "8", "--no-use_gat"])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    args = arg_parser.get_args()
    assert args.use_gat is False
