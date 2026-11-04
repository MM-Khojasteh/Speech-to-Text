"""
Environment configuration layer (dependency-free).

This module implements the project's configuration-matrix policy:

* Tuned hyperparameters, the CTC blank-penalty tier matrix, and logic toggles
  are supplied via ``STT_*`` environment variables rather than committed
  files.
* Real values live only in the local, git-ignored ``.env`` file.
* ``.env.example`` documents the full variable matrix with placeholders.

The repository therefore cannot be trained out-of-the-box: configuration
loading fails fast with a clear error listing every missing variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# Repository root (this file is src/utils/env.py -> two levels up)
_REPO_ROOT = Path(__file__).resolve().parents[2]

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}

# Tier matrix: ascending (threshold, exponent, multiplier) tuples
PenaltyTiers = List[Tuple[float, float, float]]


class MissingEnvironmentError(RuntimeError):
    """Raised when one or more required environment variables are not set."""

    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        bullet_list = "\n  - ".join(self.missing)
        super().__init__(
            "Missing required environment variables:\n"
            f"  - {bullet_list}\n\n"
            "Copy .env.example to .env at the project root and fill in your "
            "values. The .env file is deliberately excluded from version "
            "control; .env.example documents every variable."
        )


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: Optional[Path] = None) -> Optional[Path]:
    """
    Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Rules:
    * Variables already present in the environment are never overridden —
      real shell exports always win over file contents.
    * Blank lines and ``#`` comment lines are skipped; inline comments after
      an unquoted value are stripped; surrounding quotes are removed.

    Returns the path of the file that was loaded, or ``None`` if not found.
    """
    candidates = [Path(path)] if path else [Path.cwd() / ".env", _REPO_ROOT / ".env"]
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            with open(candidate, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if not key:
                        continue
                    value = value.strip()
                    if value and not value[0] in ("'", '"'):
                        value = value.split(" #", 1)[0].strip()
                    value = _strip_quotes(value)
                    os.environ.setdefault(key, value)
            return candidate
        except OSError:
            continue
    return None


def to_bool(raw: str) -> bool:
    """Parse a boolean environment value; raises ``ValueError`` on garbage."""
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"cannot interpret {raw!r} as a boolean")


def require_env(name: str, cast: Callable[[str], object] = str) -> object:
    """
    Read a REQUIRED environment variable and cast it.

    Raises :class:`MissingEnvironmentError` when the variable is unset or
    blank, and ``ValueError`` when the value cannot be cast.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        raise MissingEnvironmentError([name])
    try:
        return cast(raw.strip())
    except ValueError as error:
        raise ValueError(f"Invalid value for {name}: {raw!r} ({error})") from error


def optional_env(
    name: str, default: object, cast: Callable[[str], object] = str
) -> object:
    """
    Read an OPTIONAL environment variable, falling back to ``default``.

    An unset or blank variable yields the default; a present-but-invalid
    value still raises (a bad explicit value should never fail silently).
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return require_env(name, cast)


class EnvSource:
    """
    Collector that gathers every required variable before failing.

    Use when many variables are required at once, so the error message lists
    ALL missing names in a single actionable message instead of failing on
    the first one.
    """

    def __init__(self) -> None:
        self._missing: List[str] = []

    def require(self, name: str, cast: Callable[[str], object] = str) -> Optional[object]:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            self._missing.append(name)
            return None
        try:
            return cast(raw.strip())
        except ValueError as error:
            raise ValueError(f"Invalid value for {name}: {raw!r} ({error})") from error

    def finalize(self) -> None:
        """Raise once, listing everything that is missing."""
        if self._missing:
            raise MissingEnvironmentError(self._missing)


def parse_penalty_tiers(spec: str) -> PenaltyTiers:
    """
    Parse the blank-penalty tier matrix from a string.

    Format: ``threshold:exponent:multiplier`` segments separated by commas,
    with strictly ascending thresholds, e.g.::

        0.0:1:1.0,0.5:2:2.0

    For each sample, the highest tier whose threshold is exceeded by the
    average blank probability supplies ``probability ** exponent *
    multiplier``.
    """
    tiers: PenaltyTiers = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Malformed penalty tier {chunk!r}: "
                "expected 'threshold:exponent:multiplier'"
            )
        threshold, exponent, multiplier = (float(part) for part in parts)
        tiers.append((threshold, exponent, multiplier))
    if not tiers:
        raise ValueError("Penalty tier specification is empty")
    for lower, upper in zip(tiers, tiers[1:]):
        if upper[0] <= lower[0]:
            raise ValueError(
                f"Penalty tiers must have strictly ascending thresholds: {spec!r}"
            )
    return tiers
