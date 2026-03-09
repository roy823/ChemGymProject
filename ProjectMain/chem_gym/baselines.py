import math
from typing import Dict, List
import numpy as np
from chem_gym.envs.chem_env import ChemGymEnv

def random_search(env: ChemGymEnv, total_steps: int) -> Dict:
    """
    随机搜索：通过随机交换两个不同元素的位置来优化。
    """
    best_energy = math.inf
    history = []
    env.reset()
    n_swaps = total_steps // 2
    
    for _ in range(n_swaps):
        idx_i, idx_j = np.random.choice(env.n_active_atoms, 2, replace=False)
        elem_i = env.state[idx_i]
        elem_j = env.state[idx_j]
        
        if elem_i == elem_j:
            history.extend([best_energy, best_energy])
            continue
        env.step(idx_i * env.n_elements + elem_j)
        _, _, _, _, info = env.step(idx_j * env.n_elements + elem_i)
        
        current_energy = info["energy"]
        if current_energy < best_energy:
            best_energy = current_energy
            
        history.extend([best_energy, best_energy])
        
    return {"best_energy": best_energy, "history": history[:total_steps]}

def simulated_annealing(env: ChemGymEnv, total_steps: int, t_start: float = 0.1, t_end: float = 0.001) -> Dict:
    """
    模拟退火：交换的 Metropolis 采样。
    """
    _, info = env.reset()
    current_energy = info["energy"]
    best_energy = current_energy
    history = []

    n_swaps = total_steps // 2

    for step in range(n_swaps):
        temp = t_start * (t_end / t_start) ** (step / n_swaps)
        
        #备份
        old_state = env.state.copy()
        old_energy = current_energy

        idx_i, idx_j = np.random.choice(env.n_active_atoms, 2, replace=False)
        elem_i = old_state[idx_i]
        elem_j = old_state[idx_j]
        
        if elem_i == elem_j:
            history.extend([best_energy, best_energy])
            continue

        env.step(idx_i * env.n_elements + elem_j)
        _, _, _, _, info = env.step(idx_j * env.n_elements + elem_i)
        new_energy = info["energy"]
        
        delta_e = new_energy - old_energy
        
        #Metropolis
        if delta_e < 0 or math.exp(-delta_e / max(temp, 1e-8)) > np.random.rand():
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
        else:
            env.step(idx_i * env.n_elements + elem_i)
            env.step(idx_j * env.n_elements + elem_j)

        history.extend([best_energy, best_energy])
        
    return {"best_energy": best_energy, "history": history[:total_steps]}