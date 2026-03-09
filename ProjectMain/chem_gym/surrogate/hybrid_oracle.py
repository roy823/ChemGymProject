from __future__ import annotations

from typing import Optional

from ase import Atoms

from chem_gym.surrogate.ocp_model import EquiformerV2Oracle, OC25EnsembleOracle, UMAOracle


class HybridGrandPotentialOracle:
    """
    Hybrid oracle for operando alloy simulations:
    - clean slab energy from UMA (OMat24 formation-energy-calibrated model)
    - adsorbate-containing system energy from EquiformerV2 (S2EF)

    This follows the common grand-potential workflow where slab and adsorbate
    effects can be evaluated by separate high-fidelity models, then recombined.
    """

    def __init__(
        self,
        slab_oracle: Optional[UMAOracle] = None,
        ads_oracle: Optional[EquiformerV2Oracle | OC25EnsembleOracle] = None,
    ) -> None:
        self.slab_oracle = slab_oracle
        self.ads_oracle = ads_oracle

    def compute_slab_energy(self, atoms: Atoms, relax: bool = False) -> float:
        if self.slab_oracle is not None:
            if hasattr(self.slab_oracle, "compute_total_energy"):
                return float(self.slab_oracle.compute_total_energy(atoms, relax=relax))
            # Backward-compat fallback: legacy UMA compute_energy may be per-atom
            e = float(self.slab_oracle.compute_energy(atoms, relax=relax))
            if abs(e) < 20.0:
                return e * len(atoms)
            return e

        if self.ads_oracle is not None:
            return float(self.ads_oracle.compute_energy(atoms, relax=relax))

        return 0.0

    def compute_adsorbate_system_energy(self, atoms: Atoms, relax: bool = False) -> float:
        if self.ads_oracle is not None:
            return float(self.ads_oracle.compute_energy(atoms, relax=relax))

        if self.slab_oracle is not None:
            if hasattr(self.slab_oracle, "compute_total_energy"):
                return float(self.slab_oracle.compute_total_energy(atoms, relax=relax))
            e = float(self.slab_oracle.compute_energy(atoms, relax=relax))
            if abs(e) < 20.0:
                return e * len(atoms)
            return e

        return 0.0

    def evaluate_adsorbate_system_energy(self, atoms: Atoms, relax: bool = False) -> tuple[float, float]:
        if self.ads_oracle is not None and hasattr(self.ads_oracle, "evaluate_energy"):
            mean_e, std_e = self.ads_oracle.evaluate_energy(atoms, relax=relax)
            return float(mean_e), float(std_e)
        return float(self.compute_adsorbate_system_energy(atoms, relax=relax)), 0.0

    def compute_adsorption_energy(
        self,
        slab_atoms: Atoms,
        ads_atoms: Atoms,
        n_co: int,
        co_reference_energy: float = 0.0,
        relax: bool = False,
        precomputed_ads_system_energy: Optional[float] = None,
        precomputed_slab_reference_energy: Optional[float] = None,
    ) -> float:
        """
        Compute adsorption energy in a single-model-consistent way:
            E_ads = E(slab+CO) - E(slab) - N_CO * E_CO_ref

        For hybrid mode, this uses the ads-oracle for both terms to avoid
        cross-model reference mismatch.
        """
        n_co = int(max(0, n_co))
        e_ref = float(co_reference_energy)

        if self.ads_oracle is not None:
            if precomputed_ads_system_energy is None:
                e_ads_sys = float(self.ads_oracle.compute_energy(ads_atoms, relax=relax))
            else:
                e_ads_sys = float(precomputed_ads_system_energy)

            if precomputed_slab_reference_energy is None:
                e_slab_ads_ref = float(self.ads_oracle.compute_energy(slab_atoms, relax=False))
            else:
                e_slab_ads_ref = float(precomputed_slab_reference_energy)
            return float(e_ads_sys - e_slab_ads_ref - n_co * e_ref)

        # Fallback when ads-oracle is unavailable: compute with slab oracle.
        if self.slab_oracle is not None:
            if hasattr(self.slab_oracle, "compute_total_energy"):
                e_ads_sys = float(self.slab_oracle.compute_total_energy(ads_atoms, relax=relax))
                e_slab_ref = float(self.slab_oracle.compute_total_energy(slab_atoms, relax=False))
            else:
                e_ads_sys = float(self.slab_oracle.compute_energy(ads_atoms, relax=relax))
                e_slab_ref = float(self.slab_oracle.compute_energy(slab_atoms, relax=False))
                if abs(e_ads_sys) < 20.0:
                    e_ads_sys *= len(ads_atoms)
                if abs(e_slab_ref) < 20.0:
                    e_slab_ref *= len(slab_atoms)
            return float(e_ads_sys - e_slab_ref - n_co * e_ref)

        return 0.0

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        # Generic fallback for callers that only use one method.
        return self.compute_adsorbate_system_energy(atoms, relax=relax)
