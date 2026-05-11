# End-to-End Image Transmission Framework (JSCC Simulation)

本项目是一个基于 Python 与 MATLAB 联合仿真的端到端图像传输评估框架。系统旨在量化评估主流信源编码（AV1、JPEG）在 5G NR LDPC 信道编码以及不同信道衰落条件（AWGN、OFDM+TDL多径）下的“悬崖效应”（Cliff Effect），并测定其自适应码率的性能上界（Performance Envelope），为联合信源信道编码（JSCC）及语义通信研究提供高标准的传统链路基线（SOTA Baseline）。

## 🚀 核心架构与特性

- **多模态信源编码支持**：支持 AV1 (AVIF) 与 JPEG 信源的自动压缩与比特流提取。
- **5G NR 物理层对齐**：物理层严格遵循 3GPP 5G NR 标准，包含完整的 LDPC 编译码（Base Graph 自适应）、速率匹配、交织以及 QPSK 调制。
- **全场景信道模型**：
  - **原生 AWGN**：提供无衰落的理想高斯白噪声基线。
  - **OFDM + 多径衰落 (TDL)**：支持 TDL-A/B/C/D/E 信道模型，支持自定义时延扩展（Delay Spread），用于模拟复杂城市宏蜂窝/微蜂窝环境。
- **无偏 MMSE 均衡架构 (Unbiased MMSE)**：在 OFDM 链路中采用严格的无偏 MMSE 接收机设计，消除极低信噪比下的信号振幅衰减，实现等效噪声方差的精准推导与 CSI 可信度融合，确保系统在复杂多径下逼近香农极限。
- **自动化寻优与测试引擎**：
  - **二分质量探针 (Bisection Search)**：在严格的带宽分配 ($G \times R$) 约束下，自动搜索信源编码的最佳 Quality 参数。
  - **动态扫频与柔性降级**：基于预计算的信道容量曲线 (JSON) 动态设定 SNR 扫频下界；在缺失信道文件时自动回退至 AWGN 香农极限进行安全扫频。
  - **数据白化 (Scrambling)**：采用定点伪随机扰码序列消除低码率下信源全零填充引发的 LDPC 译码相关性问题。

## 📂 项目结构

本项目按信源及物理层特性划分为三个核心仿真模块：

* `av1_ldpc_awgn_matlab/`
  * **说明**：AV1 信源在纯 AWGN 信道下的性能仿真模块。
  * **特点**：提供最纯净的香农极限对比，用于评估 AV1 + LDPC 的基础编码增益。
* `jpeg_ldpc_awgn_matlab/`
  * **说明**：JPEG 信源在纯 AWGN 信道下的性能仿真模块。
  * **特点**：与 AV1 分支完全对称，用于传统 JPEG 基线的横向对比验证。
* `av1_ldpc_ofdm_matlab/`
  * **说明**：AV1 信源在 OFDM 调制及 5G TDL 多径信道下的综合仿真模块。
  * **特点**：包含完整的时频资源映射、块衰落信道、完美信道估计 (Perfect CSI) 以及 Unbiased MMSE 软解调逻辑，代表复杂环境下的真实物理性能。
* `kodak/`
  * **说明**：标准 Kodak 图像测试数据集（.png）。
* `NTSCC-fixCBR-SNR-SSIM.xlsx`
  * **说明**：合作方提供的基于深度学习的 JSCC (NTSCC) 对比基准数据。

## 📋 环境依赖

- **操作系统**：macOS / Linux 
- **Python**：>= 3.10 (建议使用项目自带的 `venv.nosync` 虚拟环境)
- **MATLAB**：>= R2022b，需包含 Communications Toolbox 与 5G Toolbox
- **桥接接口**：安装 MATLAB Engine API for Python (`matlab.engine`)

## 🛠️ 运行指南

### 1. 启动 MATLAB 共享引擎 (必选前置步骤)
为避免 Python 每次调用时重复启动 MATLAB 进程，需开启共享引擎。
在 MATLAB 命令行窗口执行：
```matlab
matlab.engine.shareEngine('JSCC_Engine')

```

### 2. 运行仿真主程序 (以 OFDM 模块为例)

进入对应的子目录后，通过命令行参数指定信道配置及测试图像。

```bash
# 运行默认的 TDL-C 100ns 信道仿真
venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py 

# 运行自定义信道 (AWGN 或 TDL-D 30ns) 与指定图像
venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py --channel AWGN --image kodim08.png
venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py --channel TDL-D --ds 30e-9

```

*(注：`awgn` 目录下的脚本不包含 `--channel` 等参数，直接运行 `run_sim.py` 即可)*

### 3. 容量曲线预计算 (仅 OFDM 模块可选)

针对特定多径信道生成理论极限 JSON 缓存，用于加速主程序的扫频定位：

```bash
venv.nosync/bin/python av1_ldpc_ofdm_matlab/build_capacity_curve.py --channel TDL-C --ds 100e-9

```

### 4. 结果可视化

仿真完毕后，运行各目录下的可视化脚本以提取数据点并生成包络线图表：

```bash
venv.nosync/bin/python av1_ldpc_ofdm_matlab/plot_results_envelope.py

```

## ⚠️ 工程说明与限制

1. **扰码一致性**：加解扰算法的 PRNG 种子固定为 `42`，请勿在收发两端设置不同的种子参数。
2. **绝对路径约束**：脚本默认使用基于文件相对位置的绝对路径解析，请勿随意破坏各独立模块内部的目录层级依赖。