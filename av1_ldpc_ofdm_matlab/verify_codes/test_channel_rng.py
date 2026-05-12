"""
test_channel_rng.py  —  TDL 信道随机性与能量守恒蒙特卡洛探针测试
======================================================================
目标：
  1. 验证 rng('shuffle') 隔离机制是否生效（每次信道快照是否真正随机）
  2. 验证 TDL-C 信道长期统计平均能量是否严格等于 1.0（0 dB）

接口：
  sim_ofdm_worker(tx_bits_py, R, G, snr_dB, channelModel, delaySpread)
  → [ber, rx_bits, H_energy]   （H_energy 为线性值，非 dB）

运行方式：
  venv.nosync/bin/python av1_ldpc_ofdm_matlab/test_channel_rng.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 非交互后端，避免阻塞
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 路径配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG  = os.path.join(SCRIPT_DIR, "rng_test_result.png")

# ── 仿真参数 ──────────────────────────────────────────────────────────
G            = 768 * 512 * 2          # 空口总比特数 786432（与 run_sim.py 一致）
R            = 1 / 4                  # 码率 1/4 → K = 196608 bits
K            = int(G * R)             # 信源比特数
SNR_DB       = 0.0                    # 固定 0 dB
CHANNEL_TYPE = "TDL-C"
DELAY_SPREAD = 100e-9                 # 100 ns
NUM_TRIALS   = 500

# ANSI 颜色码
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# ─────────────────────────────────────────────────────────────────────
# 启动 MATLAB Engine
# ─────────────────────────────────────────────────────────────────────
print(f"{_BOLD}{'='*66}{_RESET}")
print(f"{_BOLD}  TDL 信道 RNG 隔离 & 能量守恒 蒙特卡洛探针测试{_RESET}")
print(f"{_BOLD}{'='*66}{_RESET}")
print(f"  信道模型  : {CHANNEL_TYPE}")
print(f"  时延扩展  : {DELAY_SPREAD*1e9:.0f} ns")
print(f"  固定 SNR  : {SNR_DB:+.1f} dB")
print(f"  码率 R    : 1/4  (K = {K} bits)")
print(f"  试验次数  : {NUM_TRIALS}")
print(f"{_BOLD}{'='*66}{_RESET}\n")

print("正在连接后台 MATLAB 共享引擎 (JSCC_Engine)...")
try:
    import matlab.engine
    eng = matlab.engine.connect_matlab("JSCC_Engine")
    print(f"{_GREEN}[OK]{_RESET} 已连接共享引擎 JSCC_Engine。\n")
except Exception as _e:
    print(f"{_YELLOW}[INFO]{_RESET} 未找到共享引擎，正在启动新引擎（约 20-30 秒）...")
    eng = matlab.engine.start_matlab()
    print(f"{_GREEN}[OK]{_RESET} 新引擎启动成功。\n")

eng.addpath(SCRIPT_DIR, nargout=0)

# ─────────────────────────────────────────────────────────────────────
# 构造假负载（Dummy Payload）
# ─────────────────────────────────────────────────────────────────────
rng_py = np.random.default_rng(seed=0)
tx_bits_np  = rng_py.integers(0, 2, size=K, dtype=np.int32)
tx_bits_ml  = matlab.int32(tx_bits_np.tolist())

print(f"[Payload] 随机比特流已生成：K = {K} bits  (seed=0, 固定不变)\n")

# ─────────────────────────────────────────────────────────────────────
# 蒙特卡洛主循环
# ─────────────────────────────────────────────────────────────────────
print(f"{'─'*66}")
print(f"  {'Trial':>6}  {'H_energy (linear)':>20}  {'H_energy (dB)':>14}")
print(f"{'─'*66}")

h_energy_list = []

for trial in range(NUM_TRIALS):
    _ber, _rx, H_energy_ml = eng.sim_ofdm_worker(
        tx_bits_ml,
        float(R),
        float(G),
        float(SNR_DB),
        CHANNEL_TYPE,
        float(DELAY_SPREAD),
        nargout=3
    )
    # H_energy 由 MATLAB 直接返回线性值（mean(|H|²)），无需转换
    h_lin = float(H_energy_ml)
    h_dB  = 10.0 * np.log10(h_lin) if h_lin > 0 else float("-inf")
    h_energy_list.append(h_lin)

    # 每 50 次打印一行进度
    if (trial + 1) % 50 == 0 or trial == 0:
        print(f"  {trial+1:>6}  {h_lin:>20.6f}  {h_dB:>+14.4f} dB")

print(f"{'─'*66}\n")

# ─────────────────────────────────────────────────────────────────────
# 统计分析
# ─────────────────────────────────────────────────────────────────────
h_arr        = np.array(h_energy_list, dtype=np.float64)
unique_count = len(np.unique(np.round(h_arr, decimals=10)))
mean_energy  = float(np.mean(h_arr))
variance     = float(np.var(h_arr))
h_min        = float(np.min(h_arr))
h_max        = float(np.max(h_arr))
h_arr_dB     = 10.0 * np.log10(np.clip(h_arr, 1e-30, None))

print(f"{_BOLD}{'='*66}{_RESET}")
print(f"{_BOLD}  统计分析结果（{NUM_TRIALS} 次试验）{_RESET}")
print(f"{_BOLD}{'='*66}{_RESET}")

# ── 唯一值数量（RNG 隔离检验）────────────────────────────────────────
if unique_count == 1:
    print(f"  {'唯一值数量 (Unique Count)':<28}: "
          f"{_RED}{_BOLD}{unique_count:>6}{_RESET}  "
          f"{_RED}⚠ 警报：RNG 被全局冻结！信道每次输出完全相同！{_RESET}")
else:
    print(f"  {'唯一值数量 (Unique Count)':<28}: "
          f"{_GREEN}{unique_count:>6}{_RESET}  "
          f"{_GREEN}✓ RNG 隔离正常，每次快照独立随机{_RESET}")

# ── 平均能量（能量守恒检验）──────────────────────────────────────────
mean_dB   = 10.0 * np.log10(mean_energy) if mean_energy > 0 else float("-inf")
deviation = abs(mean_energy - 1.0)
if deviation < 0.05:
    energy_flag = f"{_GREEN}✓ 能量守恒（偏差 {deviation:.4f}）{_RESET}"
elif deviation < 0.15:
    energy_flag = f"{_YELLOW}△ 轻微偏差（偏差 {deviation:.4f}）{_RESET}"
else:
    energy_flag = f"{_RED}⚠ 能量漂移严重（偏差 {deviation:.4f}）{_RESET}"

print(f"  {'平均能量 (Mean Energy)':<28}: "
      f"{mean_energy:>10.6f}  ({mean_dB:+.4f} dB)  {energy_flag}")
print(f"  {'能量方差 (Variance)':<28}: {variance:>10.6f}")
print(f"  {'最小值 (Min)':<28}: {h_min:>10.6f}  ({10*np.log10(h_min):+.4f} dB)")
print(f"  {'最大值 (Max)':<28}: {h_max:>10.6f}  ({10*np.log10(h_max):+.4f} dB)")
print(f"{_BOLD}{'='*66}{_RESET}\n")

# ─────────────────────────────────────────────────────────────────────
# 可视化输出
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"TDL-C Channel RNG & Energy Probe  "
    f"(N={NUM_TRIALS}, SNR={SNR_DB:+.0f} dB, ds={DELAY_SPREAD*1e9:.0f} ns)",
    fontsize=13, fontweight="bold"
)

# ── 图1：H_energy_dB 折线图 ───────────────────────────────────────────
ax1 = axes[0]
trial_idx = np.arange(NUM_TRIALS)
ax1.plot(trial_idx, h_arr_dB, color="#4C9BE8", linewidth=0.8,
         alpha=0.85, label="H_energy (dB)")
ax1.axhline(y=0.0, color="#E84C4C", linewidth=1.2,
            linestyle="--", label="0 dB baseline")
ax1.axhline(y=mean_dB, color="#F5A623", linewidth=1.2,
            linestyle="-.", label=f"Mean = {mean_dB:+.3f} dB")
ax1.set_xlabel("Trial Index", fontsize=11)
ax1.set_ylabel("H_energy (dB)", fontsize=11)
ax1.set_title("Per-Trial Channel Energy", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))

# ── 图2：线性 H_energy 直方图 ─────────────────────────────────────────
ax2 = axes[1]
ax2.hist(h_arr, bins=40, density=True, color="#4C9BE8",
         edgecolor="white", linewidth=0.4, alpha=0.85,
         label="H_energy PDF")
ax2.axvline(x=1.0, color="#E84C4C", linewidth=1.5,
            linestyle="--", label="Mean = 1.0 (theory)")
ax2.axvline(x=mean_energy, color="#F5A623", linewidth=1.5,
            linestyle="-.", label=f"Mean = {mean_energy:.4f} (measured)")
ax2.set_xlabel("H_energy (linear)", fontsize=11)
ax2.set_ylabel("Probability Density", fontsize=11)
ax2.set_title("H_energy Distribution (PDF)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 统计信息文本框
stats_text = (
    f"Unique: {unique_count}/{NUM_TRIALS}\n"
    f"Mean:   {mean_energy:.5f}\n"
    f"Var:    {variance:.5f}\n"
    f"Min:    {h_min:.5f}\n"
    f"Max:    {h_max:.5f}"
)
ax2.text(0.97, 0.97, stats_text,
         transform=ax2.transAxes,
         fontsize=8.5, verticalalignment="top",
         horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                   edgecolor="gray", alpha=0.85),
         fontfamily="monospace")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[可视化] 图表已保存至: {OUTPUT_PNG}\n")
print(f"{_BOLD}{'='*66}{_RESET}")
print(f"{_BOLD}  探针测试完成{_RESET}")
print(f"{_BOLD}{'='*66}{_RESET}")
