"""
utils.py — shared helpers: logging, paths, config, seeding.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime


# ── project root ─────────────────────────────────────────────────────────────

def get_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_cache_dir() -> Path:
    p = get_root() / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_processed_dir() -> Path:
    p = get_root() / "data" / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_figures_dir() -> Path:
    p = get_root() / "reports" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_outputs_dir() -> Path:
    p = get_root() / "experiments" / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_results_dir() -> Path:
    p = get_root() / "reports" / "results_tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


# ── cache key ─────────────────────────────────────────────────────────────────

def cache_key(city: str, category: str, date_str: str) -> str:
    raw = f"{city}_{category}_{date_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


# ── timestamp ─────────────────────────────────────────────────────────────────

def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
