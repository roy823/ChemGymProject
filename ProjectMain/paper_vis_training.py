import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- 配置 ---
LOG_DIR = "chem_gym_tensorboard"
OUTPUT_DIR = "paper_results/training_vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义我们要对比的实验组前缀
GROUPS = {
    "Standard PPO (Baseline)": "PPO_",
    "Maskable PPO (Ours)": "MaskablePPO_Experiment_"
}

def extract_tensorboard_data(log_dir, group_prefix):
    """从 TensorBoard 日志中提取奖励数据，支持多种 Tag 自动检测"""
    all_runs_data = []
    
    folders = [f for f in os.listdir(log_dir) if f.startswith(group_prefix)]
    print(f"🔍 正在处理 {group_prefix} 组，共找到 {len(folders)} 个实验文件夹...")
    
    # 可能的奖励标签名
    possible_tags = ['rollout/ep_rew_mean', 'eval/mean_reward', 'train/entropy_loss']
    
    for folder in folders:
        path = os.path.join(log_dir, folder)
        try:
            # 减小 size_guidance 以加快加载速度，只读取 scalars
            event_acc = EventAccumulator(path, size_guidance={'scalars': 500})
            event_acc.Reload()
            
            available_tags = event_acc.Tags()['scalars']
            
            # 自动寻找最合适的 Tag
            target_tag = None
            for tag in possible_tags:
                if tag in available_tags:
                    target_tag = tag
                    break
            
            if target_tag:
                scalar_events = event_acc.Scalars(target_tag)
                steps = [e.step for e in scalar_events]
                values = [e.value for e in scalar_events]
                
                if len(steps) > 0:
                    df = pd.DataFrame({'step': steps, 'value': values})
                    # 平滑处理
                    df['value'] = df['value'].rolling(window=10, min_periods=1).mean()
                    all_runs_data.append(df)
            else:
                # 如果没找到奖励，打印一下可用的 tag 方便调试
                if available_tags:
                    print(f"  ⚠️ {folder} 中未找到奖励 Tag。可用 Tags: {available_tags[:3]}...")
        except Exception as e:
            pass
            
    return all_runs_data

def plot_comparison():
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {"Standard PPO (Baseline)": "#d62728", "Maskable PPO (Ours)": "#2ca02c"}
    global_max_step = 0
    global_max_val = 0
    data_plotted = False

    for label, prefix in GROUPS.items():
        runs = extract_tensorboard_data(LOG_DIR, prefix)
        if not runs: 
            print(f"  ❌ {label} 组没有提取到有效数据。")
            continue
        
        data_plotted = True
        # 对齐所有实验的步数
        current_group_max_step = min([run['step'].max() for run in runs])
        global_max_step = max(global_max_step, current_group_max_step)
        
        common_steps = np.linspace(0, current_group_max_step, 100)
        
        interp_values = []
        for run in runs:
            interp_val = np.interp(common_steps, run['step'], run['value'])
            interp_values.append(interp_val)
            
        interp_values = np.array(interp_values)
        mean_vals = np.mean(interp_values, axis=0)
        std_vals = np.std(interp_values, axis=0)
        global_max_val = max(global_max_val, mean_vals.max())
        
        # 绘制
        ax.plot(common_steps, mean_vals, label=label, color=colors[label], linewidth=2.5)
        ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals, 
                        color=colors[label], alpha=0.15)

    if not data_plotted:
        print("❌ 错误：所有组都没有数据，请检查 TensorBoard 文件夹路径或 Tag 名称。")
        return

    ax.set_title("Training Convergence: Standard vs. Maskable PPO", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Total Timesteps", fontsize=14)
    ax.set_ylabel("Mean Episode Reward", fontsize=14)
    ax.legend(fontsize=12, loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 动态标注
    if global_max_step > 0:
        ax.annotate('Faster Convergence & Higher Stability', 
                    xy=(global_max_step*0.2, global_max_val*0.9), 
                    xytext=(global_max_step*0.05, global_max_val*0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_comparison_pro.png")
    print(f"✅ 训练对比图已保存至 {OUTPUT_DIR}/training_comparison_pro.png")

if __name__ == "__main__":
    plot_comparison()