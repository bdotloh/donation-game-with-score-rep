"""Noise on score transitions: g_j = (1 - eps) G_j + eps eps_up,
f_j = (1 - eps) F_j + eps eps_down (Eqs. 3-4)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Noise:
    eps: float = 0.0
    eps_up: float = 0.5
    eps_down: float = 0.5

    def __post_init__(self):
        for name in ("eps", "eps_up", "eps_down"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {value}.")

    @classmethod
    def from_noise_val(cls, noise_val):
        """The NOISE_VAL of helper_funcs.jl: (1 - NOISE_VAL) * trans + NOISE_VAL / 2."""
        return cls(eps=noise_val, eps_up=0.5, eps_down=0.5)
