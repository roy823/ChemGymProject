from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase import Atoms
from ase.build import fcc111

from main import maybe_load_oracle


def build_hybrid_oracle(
    ads_task: str,
    ads_sm_ckpt: str,
    ads_md_ckpt: str,
    eq2_ckpt: str,
    uma_ckpt: str,
):
    args = Namespace(
        oracle_ckpt=None,
        oracle_mode="hybrid",
        ads_task=ads_task,
        disable_ads_ensemble=False,
        ads_sm_ckpt=ads_sm_ckpt,
        ads_md_ckpt=ads_md_ckpt,
        eq2_ckpt=eq2_ckpt,
        uma_ckpt=uma_ckpt,
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=False,
    )
    oracle = maybe_load_oracle(args)
    if oracle is None:
        raise RuntimeError("Hybrid oracle unavailable.")
    return oracle


def make_slab(symbol: str, size=(4, 4, 4), a=3.615, vacuum=10.0) -> Atoms:
    slab = fcc111(symbol, size=size, a=a, vacuum=vacuum)
    slab.set_pbc(True)
    return slab


def add_co_atop(slab: Atoms, site_idx: int, site_height: float = 1.85, bond_length: float = 1.15) -> Atoms:
    atoms = slab.copy()
    site_pos = atoms.positions[site_idx]
    c_pos = np.array([site_pos[0], site_pos[1], site_pos[2] + site_height], dtype=float)
    o_pos = np.array([site_pos[0], site_pos[1], site_pos[2] + site_height + bond_length], dtype=float)
    atoms += Atoms("CO", positions=[c_pos, o_pos], cell=atoms.cell, pbc=atoms.pbc)
    return atoms


def top_surface_indices(atoms: Atoms, tol=0.35) -> List[int]:
    z = atoms.get_positions()[:, 2]
    zmax = float(np.max(z))
    return [int(i) for i, zi in enumerate(z) if zi >= zmax - tol]


def substitute_one_pd(atoms: Atoms, idx: int) -> Atoms:
    b = atoms.copy()
    syms = b.get_chemical_symbols()
    syms[int(idx)] = "Pd"
    b.set_chemical_symbols(syms)
    return b


def main():
    save_dir = Path("ProjectMain/checkpoints/week3_oracle_calibration_v1")
    save_dir.mkdir(parents=True, exist_ok=True)

    oracle = build_hybrid_oracle(
        ads_task="oc25",
        ads_sm_ckpt="ProjectMain/checkpoints/esen_sm_conserve.pt",
        ads_md_ckpt="ProjectMain/checkpoints/esen_md_direct.pt",
        eq2_ckpt="ProjectMain/checkpoints/eq2_83M_2M.pt",
        uma_ckpt="ProjectMain/checkpoints/uma-m-1p1.pt",
    )

    # Literature references used for calibration targets:
    # - Cu(111) CO adsorption low-coverage: ~ -0.57 eV (exp), around -0.54 to -0.68 eV in DFT literature.
    # - Pd(111) CO atop adsorption: ~ -1.43 eV (DFT reference value).
    # - Pd segregation energy on Cu(111), UHV: -0.21 eV (DFT-RPBE, JPCL 2021).
    refs = {
        "Eads_CO@Cu111_target_eV": -0.57,
        "Eads_CO@Pd111_target_eV": -1.43,
        "Eseg_Pd_on_Cu111_UHV_target_eV": -0.21,
    }

    rows: List[Dict] = []

    cu_slab = make_slab("Cu", size=(4, 4, 4), a=3.615, vacuum=10.0)
    pd_slab = make_slab("Pd", size=(4, 4, 4), a=3.889, vacuum=10.0)
    cu_top = top_surface_indices(cu_slab)
    pd_top = top_surface_indices(pd_slab)

    cu_center_top = cu_top[len(cu_top) // 2]
    pd_center_top = pd_top[len(pd_top) // 2]

    cu_ads = add_co_atop(cu_slab, cu_center_top)
    pd_ads = add_co_atop(pd_slab, pd_center_top)

    # Same-backend gas-phase CO reference for absolute adsorption-energy calibration.
    co_gas = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]],
        cell=[15.0, 15.0, 15.0],
        pbc=[False, False, False],
    )
    if hasattr(oracle, "ads_oracle") and getattr(oracle, "ads_oracle") is not None:
        e_co_ref = float(oracle.ads_oracle.compute_energy(co_gas, relax=False))
    else:
        e_co_ref = float(oracle.compute_adsorbate_system_energy(co_gas, relax=False))

    eads_cu_rel = float(
        oracle.compute_adsorption_energy(
            slab_atoms=cu_slab,
            ads_atoms=cu_ads,
            n_co=1,
            co_reference_energy=0.0,
            relax=False,
        )
    )
    eads_pd_rel = float(
        oracle.compute_adsorption_energy(
            slab_atoms=pd_slab,
            ads_atoms=pd_ads,
            n_co=1,
            co_reference_energy=0.0,
            relax=False,
        )
    )
    eads_cu_abs = float(
        oracle.compute_adsorption_energy(
            slab_atoms=cu_slab,
            ads_atoms=cu_ads,
            n_co=1,
            co_reference_energy=e_co_ref,
            relax=False,
        )
    )
    eads_pd_abs = float(
        oracle.compute_adsorption_energy(
            slab_atoms=pd_slab,
            ads_atoms=pd_ads,
            n_co=1,
            co_reference_energy=e_co_ref,
            relax=False,
        )
    )
    rows.append(
        {
            "metric": "E_CO_gas_reference(same_backend)",
            "oracle_eV": e_co_ref,
            "target_eV": "NA",
            "abs_error_eV": "NA",
        }
    )
    rows.append(
        {
            "metric": "Eads_CO@Cu111_relative(co_ref=0)",
            "oracle_eV": eads_cu_rel,
            "target_eV": "relative_mu_mode_only",
            "abs_error_eV": "NA",
        }
    )
    rows.append(
        {
            "metric": "Eads_CO@Pd111_relative(co_ref=0)",
            "oracle_eV": eads_pd_rel,
            "target_eV": "relative_mu_mode_only",
            "abs_error_eV": "NA",
        }
    )
    rows.append(
        {
            "metric": "Eads_CO@Cu111_absolute",
            "oracle_eV": eads_cu_abs,
            "target_eV": refs["Eads_CO@Cu111_target_eV"],
            "abs_error_eV": abs(eads_cu_abs - refs["Eads_CO@Cu111_target_eV"]),
        }
    )
    rows.append(
        {
            "metric": "Eads_CO@Pd111_absolute",
            "oracle_eV": eads_pd_abs,
            "target_eV": refs["Eads_CO@Pd111_target_eV"],
            "abs_error_eV": abs(eads_pd_abs - refs["Eads_CO@Pd111_target_eV"]),
        }
    )

    cu_base = make_slab("Cu", size=(4, 4, 4), a=3.615, vacuum=10.0)
    top_ids = top_surface_indices(cu_base)
    top_id = top_ids[len(top_ids) // 2]
    # choose one sub-surface atom under top layer by nearest z value below top layer
    z = cu_base.get_positions()[:, 2]
    z_top = float(np.max(z))
    z_sub = float(np.partition(np.unique(np.round(z, 6)), -2)[-2])
    sub_ids = [int(i) for i, zi in enumerate(z) if abs(zi - z_sub) < 1e-4]
    sub_id = sub_ids[len(sub_ids) // 2]

    cu_pd_surface = substitute_one_pd(cu_base, top_id)
    cu_pd_sub = substitute_one_pd(cu_base, sub_id)

    e_surface = float(oracle.compute_slab_energy(cu_pd_surface, relax=False))
    e_sub = float(oracle.compute_slab_energy(cu_pd_sub, relax=False))
    eseg_vac = e_surface - e_sub
    rows.append(
        {
            "metric": "Eseg_Pd@Cu111_UHV(surface-subsurface)",
            "oracle_eV": eseg_vac,
            "target_eV": refs["Eseg_Pd_on_Cu111_UHV_target_eV"],
            "abs_error_eV": abs(eseg_vac - refs["Eseg_Pd_on_Cu111_UHV_target_eV"]),
        }
    )

    # CO-induced segregation tendency (qualitative check): should make surface Pd more favorable.
    surf_ads = add_co_atop(cu_pd_surface, top_id)
    sub_ads = add_co_atop(cu_pd_sub, top_id)
    e_surface_co = float(
        oracle.compute_adsorption_energy(
            slab_atoms=cu_pd_surface,
            ads_atoms=surf_ads,
            n_co=1,
            co_reference_energy=0.0,
            relax=False,
        )
        + oracle.compute_slab_energy(cu_pd_surface, relax=False)
    )
    e_sub_co = float(
        oracle.compute_adsorption_energy(
            slab_atoms=cu_pd_sub,
            ads_atoms=sub_ads,
            n_co=1,
            co_reference_energy=0.0,
            relax=False,
        )
        + oracle.compute_slab_energy(cu_pd_sub, relax=False)
    )
    eseg_co = e_surface_co - e_sub_co
    rows.append(
        {
            "metric": "Eseg_Pd@Cu111_with_CO(surface-subsurface)",
            "oracle_eV": eseg_co,
            "target_eV": "qualitative: lower_than_UHV",
            "abs_error_eV": "NA",
        }
    )
    rows.append(
        {
            "metric": "Delta_Eseg(CO-UHV)",
            "oracle_eV": eseg_co - eseg_vac,
            "target_eV": "qualitative: negative",
            "abs_error_eV": "NA",
        }
    )

    csv_path = save_dir / "calibration_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "oracle_eV", "target_eV", "abs_error_eV"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("CALIBRATION_RESULTS")
    for r in rows:
        print(r)
    print(f"CSV_SAVED {csv_path}")


if __name__ == "__main__":
    main()
