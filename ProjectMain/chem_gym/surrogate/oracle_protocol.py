"""
Unified oracle protocol for energy evaluation backends.

All oracle implementations should conform to EnergyOracle so that
ChemGymEnv can dispatch energy calls without try/except probing.
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

from ase import Atoms


@runtime_checkable
class EnergyOracle(Protocol):
    """Minimal protocol every energy oracle must satisfy."""

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        """Return total energy (eV) for the given structure."""
        ...


@runtime_checkable
class SlabOracle(EnergyOracle, Protocol):
    """Oracle capable of evaluating clean-slab energies."""

    def compute_slab_energy(self, atoms: Atoms, relax: bool = False) -> float: ...


@runtime_checkable
class AdsorbateOracle(EnergyOracle, Protocol):
    """Oracle capable of evaluating adsorbate-containing systems."""

    def compute_adsorbate_system_energy(self, atoms: Atoms, relax: bool = False) -> float: ...


@runtime_checkable
class UncertaintyOracle(EnergyOracle, Protocol):
    """Oracle that additionally reports prediction uncertainty."""

    def evaluate_energy(self, atoms: Atoms, relax: bool = False) -> Tuple[float, float]:
        """Return (mean_energy, std_uncertainty)."""
        ...


@runtime_checkable
class AdsorptionEnergyOracle(EnergyOracle, Protocol):
    """Oracle that can directly compute adsorption energy."""

    def compute_adsorption_energy(
        self,
        slab_atoms: Atoms,
        ads_atoms: Atoms,
        n_co: int,
        co_reference_energy: float = 0.0,
        relax: bool = False,
        precomputed_ads_system_energy: Optional[float] = None,
    ) -> float: ...


def evaluate_with_oracle(
    oracle: EnergyOracle,
    atoms: Atoms,
    role: str,
) -> Tuple[float, float, str]:
    """
    Dispatch an energy evaluation to the appropriate oracle method based on
    the role, returning (energy, uncertainty, backend_tag).

    This replaces the cascading try/except pattern in ChemGymEnv._evaluate_energy.
    """
    # Role-specific dispatch
    if role == "slab" and isinstance(oracle, SlabOracle):
        energy = float(oracle.compute_slab_energy(atoms, relax=False))
        return energy, 0.0, "oracle:compute_slab_energy"

    if role in {"ads_system", "ads_reference"}:
        if isinstance(oracle, UncertaintyOracle):
            # Prefer uncertainty-aware method
            if isinstance(oracle, AdsorbateOracle):
                mean_e, std_e = oracle.evaluate_energy(atoms, relax=False)
                return float(mean_e), float(std_e), "oracle:evaluate_energy"
        if isinstance(oracle, AdsorbateOracle):
            energy = float(oracle.compute_adsorbate_system_energy(atoms, relax=False))
            return energy, 0.0, "oracle:compute_adsorbate_system_energy"

    # Generic fallback: uncertainty-aware if available, otherwise plain energy
    if isinstance(oracle, UncertaintyOracle):
        mean_e, std_e = oracle.evaluate_energy(atoms, relax=False)
        return float(mean_e), float(std_e), "oracle:evaluate_energy"

    energy = float(oracle.compute_energy(atoms, relax=False))
    return energy, 0.0, "oracle:compute_energy"
