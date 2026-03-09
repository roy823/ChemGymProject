from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.optimize import LBFGS

try:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator, FormationEnergyCalculator
except Exception:  # pragma: no cover
    FAIRChemCalculator = None
    FormationEnergyCalculator = None

try:
    # Legacy OCP/OCPModels path (older EquiformerV2 workflows)
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
except Exception:  # pragma: no cover
    try:
        from ocpmodels.common.relaxation.ase_utils import OCPCalculator
    except Exception:  # pragma: no cover
        OCPCalculator = None


def _pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


class FAIRChemTaskOracle:
    """
    Generic FAIRChem oracle for one task (e.g. oc25 / oc20 / omat).
    """

    def __init__(self, checkpoint_path: str, task_name: str = "oc25", device: str = "cuda"):
        if FAIRChemCalculator is None:
            raise ImportError("FAIRChemCalculator is unavailable in current environment.")

        self.device = _pick_device(device)
        self.task_name = str(task_name)
        self.checkpoint_path = str(checkpoint_path)

        try:
            self.calculator = FAIRChemCalculator.from_model_checkpoint(
                self.checkpoint_path,
                task_name=self.task_name,
                device=self.device,
            )
        except TypeError:
            # Backward compatibility with older function signatures
            self.calculator = FAIRChemCalculator.from_model_checkpoint(
                checkpoint=self.checkpoint_path,
                task_name=self.task_name,
                device=self.device,
            )

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        atoms_calc = deepcopy(atoms)
        atoms_calc.calc = self.calculator

        if relax:
            # Only used when model provides reliable forces.
            try:
                opt = LBFGS(atoms_calc, logfile=None)
                opt.run(fmax=0.05, steps=50)
            except Exception:
                pass

        return float(atoms_calc.get_potential_energy())

    def compute_adsorbate_system_energy(self, atoms: Atoms, relax: bool = False) -> float:
        return self.compute_energy(atoms, relax=relax)


class OC25EnsembleOracle:
    """
    Ensemble oracle for OC25-like checkpoints.
    Uses mean prediction to reduce single-checkpoint noise/bias.
    """

    def __init__(
        self,
        checkpoint_paths: Sequence[str],
        task_name: str = "oc25",
        device: str = "cuda",
    ):
        paths = [str(p) for p in checkpoint_paths if p]
        if not paths:
            raise ValueError("OC25EnsembleOracle requires at least one checkpoint path.")

        self.members: List[FAIRChemTaskOracle] = [
            FAIRChemTaskOracle(p, task_name=task_name, device=device) for p in paths
        ]

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        mean_e, _ = self.evaluate_energy(atoms, relax=relax)
        return mean_e

    def compute_adsorbate_system_energy(self, atoms: Atoms, relax: bool = False) -> float:
        return self.compute_energy(atoms, relax=relax)

    def evaluate_energy(self, atoms: Atoms, relax: bool = False) -> tuple[float, float]:
        energies = [m.compute_energy(atoms, relax=relax) for m in self.members]
        energies_arr = np.asarray(energies, dtype=float)
        return float(np.mean(energies_arr)), float(np.std(energies_arr))


class EquiformerV2Oracle:
    """
    Backward-compatible adsorbate oracle.

    Loading priority:
    1) legacy OCPCalculator (if available and checkpoint is compatible)
    2) FAIRChemCalculator with configurable task (default: oc25)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        fmax: float = 0.05,
        max_steps: int = 100,
        task_name: str = "oc25",
    ):
        self.device = _pick_device(device)
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.task_name = str(task_name)
        self.checkpoint_path = str(checkpoint_path)

        self.calculator = None
        self.backend = None

        # Try legacy loader first for old eq2 checkpoints.
        if OCPCalculator is not None:
            try:
                self.calculator = OCPCalculator(
                    checkpoint=self.checkpoint_path,
                    cpu=(self.device == "cpu"),
                )
                self.backend = "legacy-ocpcalculator"
            except Exception:
                try:
                    self.calculator = OCPCalculator(
                        checkpoint_path=self.checkpoint_path,
                        cpu=(self.device == "cpu"),
                    )
                    self.backend = "legacy-ocpcalculator"
                except Exception:
                    self.calculator = None
                    self.backend = None

        # Fallback to fairchem task-based calculator.
        if self.calculator is None:
            if FAIRChemCalculator is None:
                raise ImportError(
                    "Neither OCPCalculator nor FAIRChemCalculator is available for ads oracle."
                )
            try:
                self.calculator = FAIRChemCalculator.from_model_checkpoint(
                    self.checkpoint_path,
                    task_name=self.task_name,
                    device=self.device,
                )
            except TypeError:
                self.calculator = FAIRChemCalculator.from_model_checkpoint(
                    checkpoint=self.checkpoint_path,
                    task_name=self.task_name,
                    device=self.device,
                )
            self.backend = f"fairchem-{self.task_name}"

        print(f"[AdsOracle] Loaded {Path(self.checkpoint_path).name} via {self.backend}")

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        atoms_calc = deepcopy(atoms)
        atoms_calc.calc = self.calculator

        if relax:
            try:
                opt = LBFGS(atoms_calc, logfile=None)
                opt.run(fmax=self.fmax, steps=self.max_steps)
            except Exception:
                pass

        return float(atoms_calc.get_potential_energy())

    def compute_adsorbate_system_energy(self, atoms: Atoms, relax: bool = False) -> float:
        return self.compute_energy(atoms, relax=relax)


class UMAOracle:
    """
    UMA (OMat) oracle for slab/composition thermodynamics.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        if FAIRChemCalculator is None:
            raise ImportError("FAIRChemCalculator is unavailable in current environment.")

        self.device = _pick_device(device)
        self.checkpoint_path = str(checkpoint_path)

        print(f"[UMA] Loading checkpoint from {self.checkpoint_path} ...")

        try:
            self.base_calc = FAIRChemCalculator.from_model_checkpoint(
                self.checkpoint_path,
                task_name="omat",
                device=self.device,
            )
        except TypeError:
            self.base_calc = FAIRChemCalculator.from_model_checkpoint(
                checkpoint=self.checkpoint_path,
                task_name="omat",
                device=self.device,
            )

        self.calculator = None
        if FormationEnergyCalculator is not None:
            try:
                self.calculator = FormationEnergyCalculator(self.base_calc, apply_corrections=False)
                print("[UMA] FormationEnergyCalculator initialized.")
            except Exception:
                self.calculator = None

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        """
        Return per-atom formation-like energy for compatibility.
        """
        atoms_calc = deepcopy(atoms)
        try:
            if self.calculator is not None:
                atoms_calc.calc = self.calculator
                total_form_e = float(atoms_calc.get_potential_energy())
                return total_form_e / max(1, len(atoms_calc))
        except Exception:
            pass

        atoms_calc.calc = self.base_calc
        total_e = float(atoms_calc.get_potential_energy())
        return total_e / max(1, len(atoms_calc))

    def compute_total_energy(self, atoms: Atoms, relax: bool = False) -> float:
        return float(self.compute_energy(atoms, relax=relax) * len(atoms))
