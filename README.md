# ChemGymProject 使用说明

本项目是一个将强化学习（PPO / MaskablePPO）与材料结构能量评估（UMA / Equiformer 路线）结合的实验代码库。

当前仓库的实际代码入口在 `ProjectMain/main.py`，不是根目录的 `main.py`。

## 1. 项目结构

```text
ChemGymProject/
├─ ProjectMain/
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ checkpoints/
│  │  ├─ uma-s-1p1.pt
│  │  └─ references/
│  │     ├─ form_elem_refs.yaml
│  │     └─ iso_atom_elem_refs.yaml
│  └─ chem_gym/
│     ├─ envs/chem_env.py
│     ├─ agent/trainer.py
│     ├─ surrogate/ocp_model.py
│     └─ analysis/
└─ README.md
```

## 2. 环境准备

推荐使用 Conda，Python 3.11。

### 2.1 创建环境

```bash
conda create -n chemgym python=3.11 -y
conda activate chemgym
```

### 2.2 安装 PyTorch（先装）

GPU（CUDA 12.8）：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

CPU：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 安装项目依赖

```bash
pip install -r ProjectMain/requirements.txt
```

### 2.4 验证安装

```bash
python -m pip check
python ProjectMain/main.py --help
```

## 3. 启动前注意事项

1. 请在仓库根目录运行命令：`ChemGymProject/`
2. 训练命令中的权重路径应写成 `ProjectMain/checkpoints/...`
3. Windows 下建议设置无界面后端，避免 Tk 相关崩溃：

PowerShell:

```powershell
$env:MPLBACKEND='Agg'
```

Linux/macOS:

```bash
export MPLBACKEND=Agg
```

## 4. 训练模型

### 4.1 MaskablePPO（推荐）

在仓库根目录运行：

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode graph \
  --total-steps 40000 \
  --n-active-layers 3 \
  --learning-rate 1e-4 \
  --device cuda \
  --use-masking
```

### 4.2 标准 PPO

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode graph \
  --total-steps 40000 \
  --n-active-layers 3 \
  --learning-rate 1e-4 \
  --device cuda
```

### 4.3 快速冒烟测试（推荐先跑）

```bash
python ProjectMain/main.py \
  --mode train \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --obs-mode image \
  --total-steps 2048 \
  --n-active-layers 2 \
  --learning-rate 1e-4 \
  --device cpu \
  --use-masking \
  --save-dir ProjectMain/checkpoints/smoke_test
```

说明：

- `trainer.py` 中 `n_steps=2048` 固定，因此训练步数会按 2048 的 rollout 进行对齐。
- 即使你传入较小 `--total-steps`，实际可能至少跑完一轮 2048。

## 5. 评估模型

```bash
python ProjectMain/main.py \
  --mode eval \
  --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt \
  --load-dir ProjectMain/checkpoints/<your_run_dir> \
  --use-masking
```

评估输出：

- `best_optimized.xyz`
- 可视化文件（html/png/cif，取决于回调和配置）

## 6. TensorBoard

训练日志目录固定为仓库根目录下：

```text
./chem_gym_tensorboard/
```

启动：

```bash
tensorboard --logdir ./chem_gym_tensorboard/
```

## 7. 运行产物位置

默认保存目录是 `checkpoints`（相对你执行命令时的当前路径）。

建议显式指定：

```bash
--save-dir ProjectMain/checkpoints
```

每次运行会创建类似目录：

```text
ProjectMain/checkpoints/maskable_40k_0223_1234/
├─ model.zip
├─ vec_normalize.pkl
└─ vis/
```

## 8. 平台命令示例

### 8.1 Windows PowerShell

```powershell
conda activate chemgym
cd d:\学习相关文档\科研训练\ChemGymProject
$env:MPLBACKEND='Agg'
python ProjectMain\main.py --mode train --oracle-ckpt ProjectMain\checkpoints\uma-s-1p1.pt --obs-mode graph --total-steps 40000 --n-active-layers 3 --learning-rate 1e-4 --device cuda --use-masking
```

### 8.2 Linux 服务器

```bash
conda activate chemgym
cd /root/shared-nvme/ChemGymProject
export MPLBACKEND=Agg
python ProjectMain/main.py --mode train --oracle-ckpt ProjectMain/checkpoints/uma-s-1p1.pt --obs-mode graph --total-steps 40000 --n-active-layers 3 --learning-rate 1e-4 --device cuda --use-masking
```

## 9. 已知限制与常见问题

### 9.1 `Tcl_AsyncDelete` / Tk 线程错误（Windows）

现象：训练跑一段后报 Tk 相关异常退出。  
原因：可视化回调使用 matplotlib 图形后端，在 Windows 多线程/回调场景下不稳定。  
处理：运行前设置 `MPLBACKEND=Agg`。

### 9.2 路径报错（找不到 `main.py` 或 checkpoint）

请确认你在仓库根目录，并使用：

- 入口：`ProjectMain/main.py`
- 权重：`ProjectMain/checkpoints/uma-s-1p1.pt`

### 9.3 UMA 与 Equiformer 路线

- 当前依赖栈已验证 UMA 路线可用（`FAIRChemCalculator`）。
- `OCPCalculator`（Equiformer 旧路径）在当前环境可能不可用，需要单独兼容处理。

## 10. 主要参数速查

- `--mode`：`train` / `eval` / `baseline`
- `--obs-mode`：`image` / `graph`
- `--total-steps`：总训练步数
- `--n-active-layers`：可优化的顶层层数
- `--learning-rate`：学习率
- `--device`：`cuda` / `cpu` / `auto`
- `--use-masking`：启用 MaskablePPO
- `--oracle-ckpt`：Oracle 模型权重路径
- `--save-dir`：输出目录
- `--load-dir`：评估时加载目录

---

如需进一步稳定训练（例如禁用可视化回调、修复 baseline 入口、分离 UMA/Equiformer 环境），建议先从 `ProjectMain/chem_gym/agent/trainer.py` 和 `ProjectMain/main.py` 开始调整。
