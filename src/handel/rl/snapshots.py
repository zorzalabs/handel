"""Snapshots: frozen copies of the policy that a league run saves and plays against.

A snapshot is an ordinary checkpoint (``handel-rl-eval`` can load it) stored as
``<run folder>/snapshots/snap_000050.pt``, where 50 is the update it was taken
after. In a lineup it is named ``"snap:000050"``.

Rollout workers load a snapshot from disk the first time a game needs it and
keep the most recently used ones in memory. A snapshot never changes, so
nothing has to be sent to the workers again.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .policy import NumpyPolicy, PolicyConfig, PolicyNet, read_checkpoint, save_checkpoint

PREFIX = "snap:"
CACHE_SIZE = 32  # snapshots kept in memory per process (about 1 MB each)

_cache: OrderedDict[tuple[str, int], NumpyPolicy] = OrderedDict()


def snapshot_name(update: int) -> str:
    return f"{PREFIX}{update:06d}"


def is_snapshot(name: str) -> bool:
    return name.startswith(PREFIX)


def snapshot_update(name: str) -> int:
    return int(name[len(PREFIX):])


def snapshot_path(directory: str | Path, update: int) -> Path:
    return Path(directory) / f"snap_{update:06d}.pt"


def list_snapshots(directory: str | Path) -> list[int]:
    """Updates of all snapshots in the folder, oldest first."""
    return sorted(int(p.stem.split("_")[1]) for p in Path(directory).glob("snap_*.pt"))


def save_snapshot(net: PolicyNet, directory: str | Path, update: int, meta: dict | None = None) -> Path:
    path = snapshot_path(directory, update)
    save_checkpoint(net, path, meta)
    _cache.pop((str(directory), update), None)  # never serve an older file from memory
    return path


def load_snapshot(directory: str | Path, name: str) -> NumpyPolicy:
    """The snapshot as a ``NumpyPolicy``, from memory when it was used recently."""
    key = (str(directory), snapshot_update(name))
    policy = _cache.get(key)
    if policy is None:
        ck = read_checkpoint(snapshot_path(*key))
        policy = NumpyPolicy(PolicyConfig(**ck["policy"]),
                             {k: v.numpy() for k, v in ck["state_dict"].items()})
        _cache[key] = policy
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    else:
        _cache.move_to_end(key)
    return policy
