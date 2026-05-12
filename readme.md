# End-to-End Image Transmission Framework (JSCC Simulation)

本项目是一个基于 Python 与 MATLAB 联合仿真的端到端图像传输评估框架。系统旨在量化评估主流信源编码（AV1、JPEG）在 5G NR LDPC 信道编码以及不同信道衰落条件（AWGN、OFDM+TDL多径）下的“悬崖效应”（Cliff Effect），并测定其自适应码率的性能上界（Performance Envelope），为联合信源信道编码（JSCC）及语义通信研究提供高标准的传统链路基线（SOTA Baseline）。

## 🚀 核心架构与特性

* **多模态信源编码支持**：支持 AV1 (AVIF) 与 JPEG 信源的自动压缩与比特流提取。
* **5G NR 物理层对齐**：物理层严格遵循 3GPP 5G NR 标准，包含完整的 LDPC 编译码（Base Graph 自适应）、速率匹配、符号级交织以及 QPSK 调制。
* **全场景信道与真实的块衰落模拟 (Block Fading)**：
* **原生 AWGN**：提供无衰落的理想高斯白噪声基线。
* **OFDM + 多径衰落 (TDL)**：支持 TDL-A/B/C/D/E 信道模型，支持自定义时延扩展（Delay Spread），真实复现频率选择性衰落。
* **中断容量 (Outage Capacity) 捕获**：在完全静止的多普勒配置下，系统支持单次快照的信道能量涨落监测（Lucky Draw 效应），引入**等效物理 SNR (Equivalent Physical SNR)** 概念，完美揭示突破遍历容量极限的物理本质。


* **极致严谨的物理层底层防护**：
* **无偏 MMSE 均衡架构 (Unbiased MMSE)**：在 OFDM 链路中采用严格的无偏 MMSE 接收机设计，消除极低信噪比下的信号振幅衰减，实现等效噪声方差的精准推导与 CSI 可信度融合。
* **严格的能量守恒对账**：时频域转换严格补偿非酉变换带来的功率缩放（`N0_time = noiseVar / Nfft` 精度误差 < 1%）。
* **沙盒级 RNG 状态隔离**：针对伪随机置换（如交织器固定种子 `rng(42)`）引入严密的全局随机态保存与恢复机制，确保蒙特卡洛多径信道抽样的绝对独立性。


* **自动化寻优与统计测试引擎**：
* **二分质量探针 (Bisection Search)**：在严格的带宽分配 ($G \times R$) 约束下，自动搜索信源编码的最佳 Quality 参数。
* **动态扫频与柔性降级**：基于预计算的信道容量曲线 (JSON) 动态设定 SNR 扫频下界；支持自动回退至 AWGN 香农极限进行安全扫频。
* **大数定律统计评估**：支持单 SNR 多次测试的蒙特卡洛统计，输出中断概率（BLER），并以此构建满足特定可靠性（如 90% SLA）的系统级 MS-SSIM 包络线（Reliability Envelope）。



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
* **特点**：包含完整的时频资源映射、块衰落信道随机抽样监控、完美信道估计 (Perfect CSI) 以及 Unbiased MMSE 软解调逻辑，代表复杂环境下的真实物理性能。


* `kodak/`
* **说明**：标准 Kodak 图像测试数据集（.png）。


* `NTSCC-fixCBR-SNR-SSIM.xlsx`
* **说明**：合作方提供的基于深度学习的 JSCC (NTSCC) 对比基准数据。



## 📋 环境依赖

* **操作系统**：macOS / Linux
* **Python**：>= 3.10 (建议使用项目自带的 `venv.nosync` 虚拟环境)
* **MATLAB**：>= R2022b，需包含 Communications Toolbox 与 5G Toolbox
* **桥接接口**：安装 MATLAB Engine API for Python (`matlab.engine`)

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

针对特定多径信道生成理论遍历容量 JSON 缓存，用于加速主程序的扫频定位（需注意消除 `reset` 状态锁定）：

```bash
venv.nosync/bin/python av1_ldpc_ofdm_matlab/build_capacity_curve.py --channel TDL-C --ds 100e-9

```

### 4. 结果可视化

仿真完毕后，运行各目录下的可视化脚本以提取数据点并生成满足特定统计可靠性的包络线图表：

```bash
venv.nosync/bin/python av1_ldpc_ofdm_matlab/plot_results_envelope.py

```

## ⚠️ 工程说明与限制

1. **扰码一致性与随机态隔离**：加解扰及交织算法的 PRNG 种子固定为 `42`。底层物理脚本已实现 `rng()` 状态保存与恢复机制，二次开发时请勿随意更改，以免污染信道生成的全局随机流。
2. **绝对路径约束**：脚本默认使用基于文件相对位置的绝对路径解析，请勿随意破坏各独立模块内部的目录层级依赖。
3. **等效 SNR 监控**：在多径仿真中，由于块衰落的单次能量涨落，偶尔会出现低于遍历容量极限成功解码的“悬崖点”，此为正常的 Outage Capacity 物理现象，可通过终端日志的 `Equiv_SNR` 与 `H_energy` 指标进行辅助查证。