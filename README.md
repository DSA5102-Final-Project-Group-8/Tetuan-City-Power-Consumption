# DSA5102 Final Project — Tetuan City Power Consumption

本仓库用于协作完成 DSA5102 Group 8 Final Project。为尽量减少环境差异，项目统一使用 `uv` 管理依赖；大家在自己的电脑上生成独立环境，并在本地选择对应的 Jupyter Kernel。
大家可以直接修改`DSA5102_Final_Project_Group8.ipynb`文件按照默认的结构完成notebook，
也可以独立生成文件完成task之后再整合到`DSA5102_Final_Project_Group8.ipynb`文件中。
如果觉得自己部分的默认结构不好想修改，可以直接修改标题，如果想修改别的同学的内容请与对应同学及时沟通喔。
注意：大家使用ai的时候尽可能在最后的ai使用情况注明喔，自己说总比被老师发现要好的多。

## 一、研究范围与当前分工

数据来源：[UCI — Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)

| 成员 | 主要工作 | 对应章节 | 具体内容 |
| --- | --- | --- | --- |
| WangKe | Task 1：线性回归与 EDA | 2.1–2.4；3.1（含 3.1.1–3.1.3）；3.3 中的线性模型结果；8.4–8.5 | 数据预处理、EDA、全局实验协议初稿、特征工程、Ridge、性能评估与下游任务交接文件 |
| ZhouSishuo | Task 1：非线性回归 | 3.2；3.3 中的非线性模型结果及与 Ridge 的比较 | Random Forest、Gradient Boosting、超参数调整、特征重要性、与 Ridge 比较 |
| ZengZijie | Task 2：降维 | 4.1–4.2；6.1 中 Task 2 与聚类衔接的相关内容 | PCA、Kernel PCA、维数选择、重构误差、成分解释 |
| GeYuzi | Task 3：K-means | 5.1；5.3 中的 K-means 结果 | K-means、聚类数选择、典型每日用电模式可视化 |
| FengZhengqi | Task 3：GMM 与跨任务分析 | 5.2；5.3 中的 GMM 结果及两种聚类方法比较；6.2 中的跨任务误差分析 | GMM 模型选择、聚类稳定性、与 K-means 比较、聚类类型与预测误差的关系 |


具体特征、验证方案和模型超参数仍以小组讨论及实验结果为准。

## 二、仓库结构

```text
Tetuan-City-Power-Consumption/
├── data/
│   └── Tetuan City power consumption.csv
├── handoff/              # 交接数据：由总 Notebook 生成，供队友在个人分支中使用
├── video_presentation/
├── DSA5102_Final_Project_Group8.ipynb
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

- `DSA5102_Final_Project_Group8.ipynb`：最终提交的总 Notebook（只依赖 `data/` 中的 CSV 即可从头运行）。
- `handoff/`：总 Notebook 运行时写出的交接数据（每日负荷曲线、每日天气、Task 1 全部模型的预测与指标），说明见 `handoff/README.md`。
- `data/`：项目使用的原始数据。
- `.python-version`：统一 Python 版本，目前为 Python 3.11.1。
- `pyproject.toml`：记录项目直接依赖。
- `uv.lock`：锁定完整依赖版本。
- `.venv/`：大家可以在自己电脑上生成的本地环境，不提交到 GitHub。

## 三、环境原则

GitHub 不保存或共享正在运行的 Python 环境，也不共享 Kernel。仓库只共享下面三项环境配置：

```text
.python-version
pyproject.toml
uv.lock
```

大家执行 `uv sync` 后，`uv` 会依据这些配置在本地创建自己的 `.venv`。因此大家的环境彼此独立，但依赖版本保持一致；不需要复制其他人的 `.venv`。

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

大家都需要在自己的电脑上注册一次 Kernel：

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

确实需要新库时，请大家在自己的工作分支运行：

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

## 十一、总 Notebook 合并指南

最终只提交 **总 Notebook + 数据**，老师不会拿到其他辅助文件。因此总 Notebook 必须做到：**只有 `data/` 中的 CSV，也能 Restart Kernel and Run All 从头顺序跑通**。个人模型、实验 Notebook 和辅助代码保留在各自分支，不放进 main。

### 1. 修改顺序

按成员编号依次修改总 Notebook，同一时间只有一个人在改：

1. 等上一位成员的 PR 合并后，`git pull` 最新的 `main`，再建立自己的分支（例如 `<name>/final-notebook-task2`）。
2. 只修改自己负责的章节（Task 2 → 第 4 节，Task 3 → 第 5 节，跨任务分析 → 第 6 节），不要改动别人的章节和 Setup 中已有的函数。
3. Restart Kernel and Run All，确认没有报错后再提交 PR，合并后在群里通知下一位。

`.ipynb` 是 JSON 文件，两个人同时改同一个 Notebook 时，Git 几乎无法正确合并，所以请严格按顺序进行。如果仍然遇到冲突，请不要直接“全部接受”某一方，先在群里确认。

### 2. Setup 部分怎么用

第 2 节开头的 **Setup** 已经定义了所有章节共用的内容，请直接使用，**不要重复读取或重新清洗数据**：

| 名称 | 内容 |
|---|---|
| `TRAIN_MODEL`、`SEED` | 是否重新训练（默认 `False`）；随机种子 42 |
| `data` | 清洗后的完整数据（时间索引，列名如 `temperature`、`zone_1`） |
| `development`、`holdout` | 开发期（1/1–10/18）与最终测试期（10/19–12/30） |
| `WEATHER`、`TARGETS`、`LABELS`、`COLORS` | 天气列、三个城区列、图例名称、统一配色 |
| `make_features`、`cv_splits`、`splits`、`metric_values`、`prediction_frame` | Task 1 的特征、时间序列交叉验证、评价指标、预测表格式 |

第 3 节结束时还会在内存中留下下列变量，后面的章节可以直接使用：

| 名称 | 内容 |
|---|---|
| `daily_load` | 364 天 × 72 维日负荷曲线（`zone_1_h00` … `zone_3_h23`），Task 2/3 的输入 |
| `metadata` | 每日平均天气、星期几（0 = 周一）、属于开发期还是测试期 |
| `test_predictions`、`oof` | Task 1 全部模型在测试期 / 验证折上的预测，残差 = 观测值 − 预测值 |
| `test_metrics`、`cv_all` | Task 1 各模型的测试指标 / 交叉验证指标 |

写在 Setup 里的规则：

- 新的 `import` 请加到 Setup 的 import 区（按字母顺序追加即可），不要在自己的章节里再写一遍读数据的代码。
- 只有**多个章节都会用到**的函数才放进 Setup；只在自己章节用的函数，放在本章节第一次使用它的代码格之前。
- 不要写 `from member1 import ...` 之类引用个人文件夹的代码，也不要读取 `handoff/` 或其他本地文件——提交时这些文件都不存在。
- 不要使用个人电脑的绝对路径。

### 3. 写自己章节的规则

- **变量加前缀**，避免覆盖别人的结果，例如 `pca_`、`kpca_`、`km_`、`gmm_`。不要重新给 `data`、`X`、`model`、`results` 这类通用名字赋值。
- **长时间训练**：用 `if TRAIN_MODEL:` 包住耗时的搜索，把搜索选出的超参数或结果直接写成常量（参考第 3.2 节的 `SELECTED_PARAMS`）。`TRAIN_MODEL = False` 时只用这些常量重新拟合，不读取任何模型文件。每个章节在 `False` 下最好在 3 分钟内跑完（目前第 1–3 节约 5 分钟）。
- **固定随机种子**：所有带随机性的模型都使用 `random_state=SEED`。
- **像写文章一样写 Markdown**：每一步前说明在做什么、为什么；结果后写出发现和解释，并回答本章节开头列出的问题。正文请用英文。
- 图表直接在 Notebook 中显示（`plt.show()`），不需要另存图片文件。
- 把模板里中文的提纲替换成正式内容；第 1.1 节 Executive Summary 中 “Tasks 2 and 3: main results to be added” 一行请在完成后补上自己的主要结论。

### 4. 提交 PR 前的检查清单

1. Kernel 选择 `Python (DSA5102 Group 8)`，执行 **Restart Kernel and Run All**，全程无报错。
2. 运行时临时把仓库中除 `data/` 以外的辅助文件移走（或在一个只含 Notebook 和 `data/` 的新文件夹中运行），确认确实不依赖其他文件。
3. 只改了自己的章节；输出保留在 Notebook 中，方便大家查看。
4. 如果自己的章节会产生队友需要的数据，可以在章节末尾写入 `handoff/`，并更新 `handoff/README.md`。
5. PR 说明写清楚：修改了哪些章节、运行时间、主要结论、待确认事项。

### 5. handoff 文件夹

`handoff/` 中的文件由总 Notebook 运行时生成，方便大家在**个人分支**中做实验时直接读取，不必先运行整个 Notebook。总 Notebook 本身只使用内存中的变量，不读取 `handoff/`。文件说明见 `handoff/README.md`。
