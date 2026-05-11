"""
test_phy_gap.py  —  物理层摸底测试脚本
========================================
目标：验证不同信道模型对悬崖点 Gap 的影响。
脱离 AV1 编解码，完全使用随机比特，聚焦纯物理层性能。

配置：
  R = 1/16，G = 786432

信道用例库：
  ('AWGN',  0.0)
  ('TDL-D', 30e-9)   视距多径，性能应接近 AWGN
  ('TDL-C', 100e-9)  非视距典型
  ('TDL-C', 300e-9)  非视距恶劣时延扩展

执行流程：
  对每个用例，先查容量极限 snr_lim，再从 snr_lim 起以 0.5 dB 步长向上扫频，
  一旦 ber == 0 立刻记录 cliff_snr，计算 Gap = cliff_snr - snr_lim，break。

输出：Markdown 格式对比表格。

运行方式：
    cd /Users/wtx/Developer/Semantic_Debug
    venv.nosync/bin/python av1_ldpc_ofdm_matlab/test_phy_gap.py
"""

import json
import os
import sys
import subprocess
import numpy as np

# ── 路径配置 ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VENV_PY    = os.path.join(REPO_ROOT, "venv.nosync", "bin", "python")

# ── 测试配置 ──────────────────────────────────────────────────
R = 1 / 16
G = 786432   # 768 * 512 * 2

# 信道用例库：(channelModel, delaySpread)
CHANNEL_CASES = [
    ("AWGN",  0.0),
    ("TDL-D", 30e-9),
    ("TDL-C", 100e-9),
    ("TDL-C", 300e-9),
]

# SNR 扫频最大步数（防止无限循环）
MAX_SNR_STEPS = 30   # 最多扫 15 dB


# ── 工具函数 ──────────────────────────────────────────────────
def get_curve_json_path(channel_model: str, delay_spread: float) -> str:
    """根据信道参数返回容量曲线 JSON 文件路径。"""
    if channel_model == "AWGN":
        return os.path.join(SCRIPT_DIR, "capacity_curve_AWGN.json")
    ds_ns = int(round(delay_spread * 1e9))
    return os.path.join(SCRIPT_DIR, f"capacity_curve_{channel_model}_{ds_ns}ns.json")


def ensure_capacity_curve(channel_model: str, delay_spread: float) -> None:
    """确保容量曲线 JSON 存在，不存在则调用 build_capacity_curve.py 生成。"""
    json_path = get_curve_json_path(channel_model, delay_spread)
    if os.path.exists(json_path):
        print(f"  [缓存命中] {os.path.basename(json_path)}")
        return

    print(f"  [缓存缺失] 正在生成 {os.path.basename(json_path)} ...")
    cmd = [
        VENV_PY,
        os.path.join(SCRIPT_DIR, "build_capacity_curve.py"),
        "--channel", channel_model,
        "--ds", str(delay_spread),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"build_capacity_curve.py 执行失败，信道: {channel_model}")
    print(f"  [生成完毕] {os.path.basename(json_path)}")


def load_snr_limit(channel_model: str, delay_spread: float, target_R: float) -> float:
    """从容量曲线 JSON 插值求 R 对应的 SNR 极限（dB）。"""
    json_path = get_curve_json_path(channel_model, delay_spread)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    snr_db_arr  = np.array(data["snr_db"],   dtype=np.float64)
    capacity_arr = np.array(data["capacity"], dtype=np.float64)
    target_cap   = target_R * 2.0   # QPSK：每符号 2 bit
    return float(np.interp(target_cap, capacity_arr, snr_db_arr))


# ── 启动 MATLAB Engine ────────────────────────────────────────
print("=" * 60)
print("物理层摸底测试脚本  test_phy_gap.py")
print(f"  R = 1/16 = {R:.6f},  G = {G}")
print("=" * 60)

print("\n正在启动 MATLAB Engine（共享引擎 JSCC_Engine）...")
import matlab.engine  # noqa: E402

try:
    eng = matlab.engine.connect_matlab('JSCC_Engine')
    print("已连接到后台共享引擎 JSCC_Engine。")
except Exception:
    print("未找到共享引擎，正在启动新的 MATLAB 实例（首次启动约需 30 秒）...")
    eng = matlab.engine.start_matlab()
    print("MATLAB Engine 启动成功。")

eng.addpath(SCRIPT_DIR, nargout=0)
print(f"已将 {SCRIPT_DIR} 加入 MATLAB 路径。\n")

# ── 生成随机 tx_bits（所有用例共用同一份，保证公平对比）────
K = int(G * R)
np.random.seed(2025)
tx_bits_np = np.random.randint(0, 2, K, dtype=np.int32)
tx_bits_ml = matlab.int32(tx_bits_np.tolist())
print(f"已生成随机 tx_bits：K = {K} bits\n")

# ── 主循环：遍历信道用例 ──────────────────────────────────────
results = []   # [(channel_model, delay_spread, snr_lim, cliff_snr, gap)]

for channel_model, delay_spread in CHANNEL_CASES:
    ds_str = f"{delay_spread * 1e9:.0f} ns" if delay_spread > 0 else "N/A"
    print(f"{'─'*60}")
    print(f"信道用例: {channel_model}  时延扩展: {ds_str}")

    # 1. 确保容量曲线存在
    ensure_capacity_curve(channel_model, delay_spread)

    # 2. 查出 R=1/16 对应的容量极限 SNR
    snr_lim = load_snr_limit(channel_model, delay_spread, R)
    print(f"  理论容量极限 SNR = {snr_lim:.2f} dB")

    # 3. 从 snr_lim 起以 0.5 dB 步长向上扫频
    cliff_snr = None
    for step in range(MAX_SNR_STEPS):
        snr = snr_lim + step * 0.5
        ber, _ = eng.sim_ofdm_worker(
            tx_bits_ml,
            float(R),
            float(G),
            float(snr),
            channel_model,
            float(delay_spread),
            nargout=2
        )
        ber = float(ber)
        print(f"    SNR={snr:+7.2f} dB  BER={ber:.2e}")

        if ber == 0.0:
            cliff_snr = snr
            break

    if cliff_snr is None:
        print(f"  [警告] 在 {MAX_SNR_STEPS} 步内未找到悬崖点，记录为 NaN")
        cliff_snr = float('nan')
        gap = float('nan')
    else:
        gap = cliff_snr - snr_lim
        print(f"  ✓ 悬崖点 cliff_snr = {cliff_snr:.2f} dB  Gap = {gap:.2f} dB")

    results.append((channel_model, delay_spread, snr_lim, cliff_snr, gap))

# ── 打印 Markdown 格式结果表格 ────────────────────────────────
print("\n\n" + "=" * 60)
print("物理层摸底测试结果")
print("=" * 60)
print()
print("| 信道模型 | 时延扩展 | 理论极限 (dB) | 实测悬崖点 (dB) | Gap (dB) |")
print("|----------|----------|---------------|-----------------|----------|")
for channel_model, delay_spread, snr_lim, cliff_snr, gap in results:
    ds_str = f"{delay_spread * 1e9:.0f} ns" if delay_spread > 0 else "N/A"
    if np.isnan(cliff_snr):
        cliff_str = "未收敛"
        gap_str   = "N/A"
    else:
        cliff_str = f"{cliff_snr:.2f}"
        gap_str   = f"{gap:.2f}"
    print(f"| {channel_model:<8} | {ds_str:<8} | {snr_lim:>13.2f} | {cliff_str:>15} | {gap_str:>8} |")

print()
print("注：Gap = 实测悬崖点 - 理论容量极限，反映物理层实现损耗。")
print("    Gap 越小，物理层越接近香农极限；Gap 越大，信道损伤越严重。")
