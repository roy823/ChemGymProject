import numpy as np
from ase.neighborlist import neighbor_list

def calculate_wcp(atoms, element_types):
    """
    计算 Warren-Cowley 短程有序参数 (WCP).
    alpha_{ij} = 1 - P_{ij} / c_j
    - alpha < 0: 倾向于吸引 (有序化)
    - alpha > 0: 倾向于排斥 (团聚)
    - alpha = 0: 随机分布
    """
    symbols = np.array(atoms.get_chemical_symbols())
    #分析真实原子间的 SRO
    real_mask = symbols != "Vac"
    real_symbols = symbols[real_mask]
    unique_elements = [e for e in element_types if e != "Vac"]
    
    #计算全局浓度 c_j
    counts = {e: np.sum(real_symbols == e) for e in unique_elements}
    total = len(real_symbols)
    c = {e: counts[e] / total for e in unique_elements}

    #获取近邻列表
    i, j = neighbor_list('ij', atoms, cutoff=3.0)
    
    wcp_matrix = {}
    for e_i in unique_elements:
        for e_j in unique_elements:
            # 找到所有中心原子为 e_i 的键
            indices_i = np.where(symbols[i] == e_i)[0]
            if len(indices_i) == 0: continue
            
            # 在这些键中，邻居是 e_j 的比例 P_{ij}
            neighbors_of_i = symbols[j[indices_i]]
            p_ij = np.sum(neighbors_of_i == e_j) / len(neighbors_of_i)
            
            # 计算 alpha
            alpha = 1 - (p_ij / c[e_j]) if c[e_j] > 0 else 0
            wcp_matrix[f"{e_i}-{e_j}"] = alpha
            
    return wcp_matrix

# 使用示例
# wcp = calculate_wcp(best_atoms, ["Cu", "Ni", "Co"])
# print(f"Cu-Cu WCP: {wcp['Cu-Cu']:.3f}") # 正值表示 Cu 趋向于团聚