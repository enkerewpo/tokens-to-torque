"""Terminal output helpers shared by every script in this repo.

CLI output is English on purpose — the tutorial prose is Chinese, but a terminal
full of Chinese is awkward to copy, grep and paste into an issue.

Colours follow NVIDIA green (#76B900 ≈ 256-colour 148). Honours NO_COLOR and
falls back to plain text when stdout is not a TTY.
"""
import os
import sys
import time

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str) -> str:
    return code if _TTY else ""


GREEN = _c("\033[38;5;148m")
DIM = _c("\033[2m")
BOLD = _c("\033[1m")
RED = _c("\033[31m")
YELLOW = _c("\033[33m")
CYAN = _c("\033[36m")
R = _c("\033[0m")


def step(msg: str) -> None:
    """A numbered stage of the script."""
    print(f"\n{BOLD}▸ {msg}{R}", flush=True)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{R} {msg}", flush=True)


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{R}", flush=True)


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{R} {msg}", flush=True)


def die(msg: str, hint: str = "") -> None:
    print(f"\n{RED}✗ {msg}{R}", file=sys.stderr)
    if hint:
        print(f"{DIM}{hint}{R}", file=sys.stderr)
    sys.exit(1)


def kv(key: str, value, unit: str = "") -> None:
    """Aligned key/value line for reporting numbers."""
    print(f"  {key:<26}{BOLD}{value}{R}{DIM}{(' ' + unit) if unit else ''}{R}", flush=True)


def done(msg: str = "Done") -> None:
    print(f"\n{GREEN}{BOLD}{msg}{R}\n", flush=True)


class Timer:
    """with Timer('loading model') as t: ...  →  prints elapsed on exit."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"  {DIM}{self.label}…{R}", end="", flush=True)
        return self

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def __exit__(self, *exc):
        if exc[0] is None:
            print(f"\r  {GREEN}✓{R} {self.label} {DIM}({self.elapsed:.1f}s){R}", flush=True)
        else:
            print(f"\r  {RED}✗{R} {self.label}", flush=True)
        return False
