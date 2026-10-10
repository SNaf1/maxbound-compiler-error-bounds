"""Settings for MaxBound, buildable from a dict or a JSON file."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Union

METHOD_CHOICES = ("best", "zonotope", "interval", "zonotope-separate", "interval-separate")
OUTPUT_CHOICES = ("model", "logits")
FP_CHOICES = ("auto", "float32", "float64", "real")


@dataclass
class MaxBoundConfig:
    """
    method:   "best" runs every applicable analysis and keeps, per output, the
              smallest bound (each is sound, so the smallest is too); or name one.
    output:   "model" bounds what the model returns (probabilities if it ends in
              softmax); "logits" bounds the values before a final softmax.
    fp_model: how the models' own floating-point execution is accounted for.
              "auto" uses the format the models store their weights in;
              "float32"/"float64" force one; "real" assumes exact arithmetic
              (gives the pure real-arithmetic bound, used by hand-checkable tests).
    check_runtime: refuse to run if PyTorch is set to use reduced-precision
              float32 matrix products (TF32/bfloat16), which would break A1.
    """

    method: str = "best"
    output: str = "model"
    fp_model: str = "auto"
    check_runtime: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHOD_CHOICES:
            raise ValueError(f"method must be one of {METHOD_CHOICES}")
        if self.output not in OUTPUT_CHOICES:
            raise ValueError(f"output must be one of {OUTPUT_CHOICES}")
        if self.fp_model not in FP_CHOICES:
            raise ValueError(f"fp_model must be one of {FP_CHOICES}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MaxBoundConfig":
        """Build from a dict; unknown keys raise ValueError."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**dict(d))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "MaxBoundConfig":
        """Build from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def build(cls, config: Optional[Union["MaxBoundConfig", Mapping[str, Any], str, Path]] = None,
              **overrides: Any) -> "MaxBoundConfig":
        """Combine the defaults, an optional config (object, dict or JSON path) and keyword overrides."""
        if config is None:
            base = {}
        elif isinstance(config, MaxBoundConfig):
            base = asdict(config)
        elif isinstance(config, Mapping):
            base = dict(config)
        else:
            base = asdict(cls.from_json(config))
        base.update(overrides)
        return cls.from_dict(base)
