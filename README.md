# DSA5102 Final Project — Tetuan City Power Consumption

本仓库用于协作完成 DSA5102 Group 8 Final Project。为尽量减少环境差异，项目统一使用 `uv` 管理依赖；每位成员在自己的电脑上生成独立环境，并在本地选择对应的 Jupyter Kernel。

## 一、研究范围与当前分工

数据来源：[UCI — Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)

| 成员 | 主要工作 | 具体内容 |
| --- | --- | --- |
| Member 1 | Task 1：线性回归与 EDA | 数据预处理、特征工程、Ridge、性能评估 |
| Member 2 | Task 1：非线性回归 | Random Forest、Gradient Boosting、超参数调整、特征重要性、与 Ridge 比较 |
| Member 3 | Task 2：降维 | PCA、Kernel PCA、维数选择、重构误差、成分解释 |
| Member 4 | Task 3：K-means | K-means、聚类数选择、典型每日用电模式可视化 |
| Member 5 | Task 3：GMM 与跨任务分析 | GMM 模型选择、聚类稳定性、与 K-means 比较、聚类类型与预测误差的关系 |

具体特征、验证方案和模型超参数仍以小组讨论及实验结果为准。

## 二、仓库结构

```text
Tetuan-City-Power-Consumption/
├── data/
│   └── Tetuan City power consumption.csv
├── video_presentation/
├── DSA5102_Final_Project_Group8.ipynb
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

- `DSA5102_Final_Project_Group8.ipynb`： Notebook，目前只是整体框架。
- `data/`：项目使用的原始数据。
- `.python-version`：统一 Python 版本，目前为 Python 3.11.1。
- `pyproject.toml`：记录项目直接依赖。
- `uv.lock`：锁定完整依赖版本。
- `.venv/`：每位成员在自己电脑上生成的本地环境，不提交到 GitHub。

## 三、环境原则

GitHub 不保存或共享正在运行的 Python 环境，也不共享 Kernel。仓库只共享下面三项环境配置：

```text
.python-version
pyproject.toml
uv.lock
```

每位成员执行 `uv sync` 后，`uv` 会依据这些配置在本地创建自己的 `.venv`。因此大家的环境彼此独立，但依赖版本保持一致；不需要复制其他人的 `.venv`。

## 四、首次建立本地环境

请先在终端进入本仓库根目录，并确认已经安装 `uv`：

```bash
uv --version
```

随后按锁文件生成本地环境：

```bash
uv sync --frozen
```

如果本机还没有 Python 3.11.1，可以先运行：

```bash
uv python install 3.11.1
uv sync --frozen
```

上述步骤会在当前仓库生成本地 `.venv`。一般不需要手动激活环境，后续命令可以统一通过 `uv run` 执行。

## 五、注册并选择 Jupyter Kernel

每位成员都需要在自己的电脑上注册一次 Kernel：

```bash
uv run python -m ipykernel install --user --name dsa5102-group8 --display-name "Python (DSA5102 Group 8)"
```

打开 `DSA5102_Final_Project_Group8.ipynb` 后，请在右上角选择：

```text
Python (DSA5102 Group 8)
```

如果列表中暂时没有该名称，也可以直接选择本仓库中的解释器：

```text
Windows:       .venv\Scripts\python.exe
macOS / Linux: .venv/bin/python
```

选择后，请在 Notebook 中运行：

```python
import sys

print(sys.executable)
```

输出路径应指向当前仓库的 `.venv`。如果指向系统 Python、Anaconda 或其他项目目录，说明需要重新选择 Kernel。

可以继续检查主要依赖：

```python
import numpy
import pandas
import scipy
import matplotlib
import seaborn
import sklearn

print("Environment OK")
```

## 六、拉取更新后的日常操作

拉取团队最新代码后，建议在开始运行 Notebook 前执行：

```bash
uv sync --frozen
```

如果依赖没有变化，该命令会很快完成；如果 `uv.lock` 有更新，它会相应更新本地 `.venv`。Kernel 名称通常不需要重复注册。

## 七、增加或修改依赖

确实需要新库时，请在自己的工作分支运行：

```bash
uv add <package-name>
```

随后请同时提交：

```text
pyproject.toml
uv.lock
```

为了让其他成员能够复现，烦请尽量不要只执行未记录的 `pip install`。依赖相关 Pull Request 合并后，其他成员只需再次运行 `uv sync --frozen`。

## 八、数据读取

请从仓库根目录打开项目，并在共享 Notebook 中使用相对路径：

```python
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "Tetuan City power consumption.csv"

assert DATA_PATH.exists(), f"找不到数据文件：{DATA_PATH}"
data = pd.read_csv(DATA_PATH)
```

请不要把个人电脑或 Cloud Studio 的绝对路径写入共享 Notebook。

## 九、协作约定

大家已经熟悉 GitHub，这里只保留与本项目直接相关的约定：

1. 每项工作从最新的 `main` 建立独立分支，建议命名为 `<name>/<task>`。
2. 独立实验优先放在任务 Notebook 中，确认后再整理进主 Notebook。
3. 修改主 Notebook 前请在群里简单说明，尽量避免多人同时修改同一区域。
4. Pull Request 请说明研究问题、主要改动、运行结果和待确认事项。
5. 合并前请使用正确 Kernel 重启并顺序运行相关代码，核对图表、指标和文字结论。
6. 如遇 Notebook 冲突，请先保留双方版本并共同确认，不建议直接使用“全部接受”覆盖其中一方。

环境是每位成员各自生成的；这里需要协调的主要是共享配置文件和 Notebook 内容，而不是彼此电脑上的 `.venv`。

## 十、学术规范

项目中的代码、分析和文字需要由小组成员理解并共同负责。使用生成式 AI 时，请按照课程要求在最终报告中如实说明使用目的与范围，并由成员复核相关代码、结果和表述。

以上说明用于保持项目可复现并减少协作成本。如有更合适的安排，欢迎大家讨论后共同更新。
