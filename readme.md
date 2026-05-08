# End-to-End Image/Video Transmission Test (JSCC Simulation)

本项目是一个基于 Python 和 MATLAB 联合仿真的端到端图像传输系统，主要用于评估 **AV1** 和 **JPEG** 信源编码在 **5G NR LDPC** 信道编码下的悬崖效应（Cliff Effect）及自适应码率性能上限（Envelope）。

## 🚀 核心特性

- **多编码支持**：支持 AV1 (AVIF) 和 JPEG 两种主流信源编码。
- **5G 标准对齐**：物理层采用 5G NR 标准的 LDPC 纠错码及 QPSK 调制。
- **数据白化 (Scrambling)**：内置伪随机扰码器，有效消除极低码率下全零序列导致的 LDPC 译码陷阱。
- **自动质量探针**：利用二分查找算法（Bisection Search）在给定带宽约束（G*R）下自动寻找最佳压缩质量。
- **包络线分析**：支持提取各码率临界点，绘制自适应码率调节后的性能上限曲线。

## 📋 环境要求

- **操作系统**：macOS (本项目提供的依赖文件 `requirements_mac.txt` 针对 Mac 优化)。
- **Python**：建议 3.10+。
- **MATLAB**：建议 R2022b 或更高版本，需安装 5G Toolbox。
- **Python-MATLAB Bridge**：需安装 `matlab.engine`。

## 🛠️ 配置与安装

### 1. 安装 Python 依赖
可参考setup_env.sh文件全自动安装。


### 2. 连接 VS Code 与 MATLAB (重要)

为了让 Python 能够调用 MATLAB 的物理层仿真函数，必须建立共享引擎连接：

1. 在终端安装 MATLAB Engine API。
2. 打开 MATLAB 软件。
3. 在 MATLAB 命令行窗口输入以下命令启动共享引擎：
    matlab.engine.shareEngine('JSCC_Engine')

*注意：必须先手动启动此引擎，Python 脚本 `run_sim.py` 才能成功连接。*

## 📂 项目结构

* `av1_ldpc_awgn_matlab/`：AV1 仿真核心目录，包含扫频脚本、绘图脚本及 MATLAB 物理层函数。
* `jpeg_ldpc_awgn_matlab/`：JPEG 仿真核心目录，逻辑与 AV1 分支完全对称。
* `kodak/`：存放 Kodak 标准测试数据集（.png）。
* `NTSCC-fixCBR-SNR-SSIM.xlsx`：用于对比的合作方 NTSCC Baseline 数据。

## 📊 使用方法

### 运行仿真

进入对应的编码目录（以 AV1 为例），运行主程序：

* 程序会自动在 `kodak/` 文件夹下寻找图片。
* 仿真完成后，数据将以 `.pkl` 格式落盘，并生成恢复后的图像。

### 绘制结果

运行包络线绘图脚本，直观展示性能上限：

python plot_results_envelope.py

该脚本会自动抓取最新的 `.pkl` 文件并生成带有红色虚线包络的学术级图表。

## ⚠️ 注意事项

* **扰码一致性**：加扰与解扰种子固定为 `42`，请勿随意更改以确保收发同步。
* **路径问题**：请确保在 Python 脚本所在的目录下执行命令，或正确配置 Python 搜索路径。

---

*本项目代码已托管至 GitHub，方便团队协作与版本管理。*
