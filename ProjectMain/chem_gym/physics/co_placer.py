from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from chem_gym.physics.analytical_prior import DEFAULT_PRIOR_CONSTANTS

try:
    from ase import Atoms
except ImportError:  # pragma: no cover
    Atoms = None


@dataclass
class COPlacementResult:
    atoms_with_co: "Atoms"
    n_co: int
    occupied_site_indices: List[int]
    target_n_co: int
    target_coverage: float
    mean_initial_delta_omega: float


class GreedyCOPlacer:
    """
    CO placement with a cheap grand-potential greedy criterion.

    For each candidate site i, approximate:
        DeltaOmega_i ~= E_ads(i) + E_rep(i | occupied) - mu_CO

    At each step, pick the site with minimum DeltaOmega_i and only accept if
    DeltaOmega_i < 0. This prevents deterministic "always fill Pd" behavior.
    """

    KB_EV_PER_K = 8.617333262145e-5

    def __init__(
        self,
        site_height: float = 1.85,
        bond_length: float = 1.15,
        repulsion_distance: float = 3.0,
        max_coverage: float = 1.0,
        repulsion_strength_ev: float = 0.15,
        repulsion_sigma_a: float = 2.0,
        use_langmuir_target: bool = True,
        e_cu_co: float | None = None,
        e_pd_co: float | None = None,
        adsorption_energies_ev: Dict[str, float] | None = None,
    ) -> None:
        self.site_height = float(site_height)
        self.bond_length = float(bond_length)
        self.repulsion_distance = float(repulsion_distance)
        self.max_coverage = float(max_coverage)
        self.repulsion_strength_ev = float(repulsion_strength_ev)
        self.repulsion_sigma_a = float(max(1e-6, repulsion_sigma_a))
        self.use_langmuir_target = bool(use_langmuir_target)
        self.adsorption_energies_ev = dict(adsorption_energies_ev or {})
        if e_cu_co is not None:
            self.adsorption_energies_ev["Cu"] = float(e_cu_co)
        if e_pd_co is not None:
            self.adsorption_energies_ev["Pd"] = float(e_pd_co)
        self.default_e_cu_co = float(DEFAULT_PRIOR_CONSTANTS["e_cu_co"])
        self.default_e_pd_co = float(DEFAULT_PRIOR_CONSTANTS["e_pd_co"])

    def _site_adsorption_energy(self, symbol: str) -> float:
        if symbol in self.adsorption_energies_ev:
            return float(self.adsorption_energies_ev[symbol])
        if symbol == "Pd":
            return self.default_e_pd_co
        if symbol == "Cu":
            return self.default_e_cu_co
        return -0.40

    def _estimate_target_n(
        self,
        site_symbols: Sequence[str],
        mu_co: float,
        temperature_k: float,
        n_candidates: int,
        coverage_cap_n: int,
    ) -> int:
        if not self.use_langmuir_target:
            return coverage_cap_n
        if n_candidates <= 0 or coverage_cap_n <= 0:
            return 0

        t = max(1.0, float(temperature_k))
        beta = 1.0 / (self.KB_EV_PER_K * t)

        theta_sum = 0.0
        for sym in site_symbols:
            delta0 = self._site_adsorption_energy(sym) - float(mu_co)
            theta_i = 1.0 / (1.0 + float(np.exp(np.clip(beta * delta0, -80.0, 80.0))))
            theta_sum += theta_i

        theta_avg = theta_sum / float(n_candidates)
        target_n = int(np.rint(theta_avg * n_candidates))
        return int(np.clip(target_n, 0, coverage_cap_n))

    def place(
        self,
        slab: "Atoms",
        candidate_indices: Sequence[int],
        mu_co: float,
        temperature_k: float = 300.0,
    ) -> COPlacementResult:
        if Atoms is None or slab is None:
            return COPlacementResult(
                atoms_with_co=slab,
                n_co=0,
                occupied_site_indices=[],
                target_n_co=0,
                target_coverage=0.0,
                mean_initial_delta_omega=0.0,
            )

        atoms = slab.copy()
        if len(candidate_indices) == 0 or self.max_coverage <= 0:
            return COPlacementResult(
                atoms_with_co=atoms,
                n_co=0,
                occupied_site_indices=[],
                target_n_co=0,
                target_coverage=0.0,
                mean_initial_delta_omega=0.0,
            )

        symbols = atoms.get_chemical_symbols()
        candidate_indices = list(candidate_indices)
        site_symbols = [symbols[idx] for idx in candidate_indices]

        coverage_cap_n = int(np.floor(self.max_coverage * len(candidate_indices) + 1e-8))
        if coverage_cap_n == 0 and self.max_coverage > 0:
            coverage_cap_n = 1

        target_n = self._estimate_target_n(
            site_symbols=site_symbols,
            mu_co=mu_co,
            temperature_k=temperature_k,
            n_candidates=len(candidate_indices),
            coverage_cap_n=coverage_cap_n,
        )
        max_sites = int(min(coverage_cap_n, target_n))
        if max_sites <= 0:
            return COPlacementResult(
                atoms_with_co=atoms,
                n_co=0,
                occupied_site_indices=[],
                target_n_co=target_n,
                target_coverage=float(target_n) / float(max(1, len(candidate_indices))),
                mean_initial_delta_omega=0.0,
            )

        tags = atoms.get_tags()
        if tags is None or len(tags) != len(atoms):
            tags = np.zeros(len(atoms), dtype=int)

        initial_delta_omega = [
            self._site_adsorption_energy(symbols[idx]) - float(mu_co)
            for idx in candidate_indices
        ]
        mean_initial_delta_omega = float(np.mean(initial_delta_omega)) if initial_delta_omega else 0.0

        c_positions: List[np.ndarray] = []
        occupied: List[int] = []
        occupied_set = set()

        while len(occupied) < max_sites:
            best_site = None
            best_c_pos = None
            best_delta_omega = None

            for site_idx in candidate_indices:
                if site_idx in occupied_set:
                    continue

                site_pos = atoms.positions[site_idx]
                c_pos = np.array([site_pos[0], site_pos[1], site_pos[2] + self.site_height], dtype=float)

                if c_positions and self.repulsion_distance > 0.0:
                    distances = [float(np.linalg.norm(c_pos - other)) for other in c_positions]
                    if min(distances) < self.repulsion_distance:
                        continue

                repulsion_e = 0.0
                for other in c_positions:
                    dist = float(np.linalg.norm(c_pos - other))
                    repulsion_e += self.repulsion_strength_ev * float(
                        np.exp(-((dist / self.repulsion_sigma_a) ** 2))
                    )

                site_ads_e = self._site_adsorption_energy(symbols[site_idx])
                delta_omega = site_ads_e + repulsion_e - float(mu_co)

                if best_delta_omega is None or delta_omega < best_delta_omega:
                    best_delta_omega = float(delta_omega)
                    best_site = int(site_idx)
                    best_c_pos = c_pos

            if best_site is None:
                break
            if best_delta_omega is None or best_delta_omega >= 0.0:
                break

            o_pos = np.array(
                [best_c_pos[0], best_c_pos[1], best_c_pos[2] + self.bond_length],
                dtype=float,
            )
            co = Atoms("CO", positions=[best_c_pos, o_pos], cell=atoms.cell, pbc=atoms.pbc)
            atoms += co

            tags = np.concatenate([tags, np.array([2, 2], dtype=int)])
            atoms.set_tags(tags)

            c_positions.append(best_c_pos)
            occupied.append(best_site)
            occupied_set.add(best_site)

        return COPlacementResult(
            atoms_with_co=atoms,
            n_co=len(occupied),
            occupied_site_indices=occupied,
            target_n_co=target_n,
            target_coverage=float(target_n) / float(max(1, len(candidate_indices))),
            mean_initial_delta_omega=mean_initial_delta_omega,
        )
