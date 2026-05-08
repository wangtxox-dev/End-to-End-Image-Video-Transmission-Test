"""
plot_results_envelope.py  —  自适应码率性能上限（包络线）可视化
从最新的 sim_jpeg_data_*.pkl 存档中读取仿真数据，绘制各码率曲线及其包络线。

使用方法：
    python plot_results_envelope.py

前置条件：先运行 run_sim.py 生成至少一个 .pkl 存档文件。
"""

import os
import glob
import pickle
import numpy as np
from fractions import Fraction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 全局可视化配置区 (Control Panel)
# ─────────────────────────────────────────────────────────────
# 【开关】True: 仅绘制纯净版包络图; False: 叠加合作方的 JSCC Baseline 对比图
ONLY_ORIGINAL_CLIFF = True

# 【路径配置】
# 注意：使用通配符自动匹配最新的 pkl 存档
PKL_PATTERN        = 'sim_jpeg_data_20260508_152221.pkl'
PARTNER_EXCEL_PATH = '../NTSCC-fixCBR-SNR-SSIM.xlsx'  # Excel 文件放在 Semantic_Debug 根目录

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def format_rate(r: float) -> str:
    """将浮点码率精确转为分数字符串，修复非1分子的Bug。"""
    frac = Fraction(r).limit_denominator(10000)
    if frac.denominator > 1000:
        return f"{r:.4f}"
    return f"{frac.numerator}/{frac.denominator}"


def format_rate_file(r: float) -> str:
    """将浮点码率转为对文件系统安全的名字。"""
    frac = Fraction(r).limit_denominator(10000)
    if frac.denominator > 1000:
        return f"{r:.4f}".replace('.', '_')
    return f"{frac.numerator}_{frac.denominator}"


# 中文字体支持：优先找通用 Unicode，找不到再找港版苹方、黑体，最后兜底
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────
# 智能寻路：自动定位最新 pkl 存档
# ─────────────────────────────────────────────────────────────
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
pkl_pattern = os.path.join(WORKER_DIR, PKL_PATTERN)
candidates  = glob.glob(pkl_pattern)

if not candidates:
    raise FileNotFoundError(
        "未找到任何仿真存档文件 (sim_jpeg_data_*.pkl)。\n"
        "请先运行 run_sim.py 完成仿真并生成数据文件。"
    )

# 按文件修改时间排序，取最新的一个
candidates.sort(key=os.path.getmtime, reverse=True)
latest_pkl = candidates[0]
pkl_basename = os.path.basename(latest_pkl)

print(f"[加载数据] 正在读取最新存档: {pkl_basename}")

# ─────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────
with open(latest_pkl, 'rb') as f:
    payload = pickle.load(f)

valid_rates: list  = payload['valid_rates']
results:     dict  = payload['results']
G:           int   = payload['G']
timestamp:   str   = payload.get('timestamp', 'unknown')

print(f"[加载数据] 存档时间戳: {timestamp}  |  有效码率数: {len(valid_rates)}")

# ─────────────────────────────────────────────────────────────
# 数据预处理：提取全局 SNR 极值，用于 X 轴视觉延伸
# ─────────────────────────────────────────────────────────────
all_snr_vals = []
for R in valid_rates:
    all_snr_vals.extend(results[R]['snr_list'])

global_min_snr = min(all_snr_vals)
global_max_snr = max(all_snr_vals)
print(f"[预处理] 全局 SNR 范围: [{global_min_snr:.1f}, {global_max_snr:.1f}] dB")

# ─────────────────────────────────────────────────────────────
# 精修绘图：折线图（实事求是的物理斜率）
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 0.8, len(valid_rates)))

# 用于收集各曲线的悬崖点坐标，最终连成包络线
cliff_points = []

for i, R in enumerate(valid_rates):
    d        = results[R]
    is_floor = d['is_floor']
    frac_str = format_rate(R)

    snr_arr  = np.array(d['snr_list'],  dtype=float)
    ssim_arr = np.array(d['ssim_list'], dtype=float)
    y_max    = d['y_max']

    # 向左延伸：左侧补 0（尚未突破悬崖）
    if snr_arr[0] > global_min_snr:
        snr_arr  = np.concatenate([[global_min_snr], snr_arr])
        ssim_arr = np.concatenate([[0.0], ssim_arr])

    # 向右延伸：右侧补 y_max（已完全恢复的稳态值）
    if snr_arr[-1] < global_max_snr:
        snr_arr  = np.concatenate([snr_arr, [global_max_snr]])
        ssim_arr = np.concatenate([ssim_arr, [y_max]])

    # 步骤 2：极简图例标签，只保留码率信息
    if is_floor:
        label = f"Limit R≈{frac_str}"
    else:
        label = f"R={frac_str}"

    # 使用 plot 绘制真实斜率折线，不使用 step，不使用 marker='.'
    ax.plot(snr_arr, ssim_arr,
            color=colors[i], linewidth=1.2, alpha=1,
            label=label)

    # 步骤 3：记录悬崖点坐标（ssim_arr > 0 的第一个点）
    cliff_indices = np.where(ssim_arr > 0)[0]
    if len(cliff_indices) > 0:
        cliff_idx = cliff_indices[0]
        cliff_snr = snr_arr[cliff_idx]
        cliff_points.append((cliff_snr, y_max))

# ─────────────────────────────────────────────────────────────
# 步骤 3（续）：绘制包络线
# ─────────────────────────────────────────────────────────────
if cliff_points:
    # 按 SNR 从小到大排序
    cliff_points.sort(key=lambda p: p[0])
    envelope_snr  = [p[0] for p in cliff_points]
    envelope_ssim = [p[1] for p in cliff_points]

    ax.plot(envelope_snr, envelope_ssim,
            color='red', linestyle='--', linewidth=2.5, zorder=5,
            marker='s', markersize=7,
            markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
            label='Adaptive Rate Envelope')
    print(f"[包络线] 共收集到 {len(cliff_points)} 个悬崖点，包络线已绘制。")

# ─────────────────────────────────────────────────────────────
# 附加项：合作方 JSCC Baseline 叠绘
# ─────────────────────────────────────────────────────────────
if not ONLY_ORIGINAL_CLIFF:
    try:
        partner_excel_abs = os.path.join(WORKER_DIR, PARTNER_EXCEL_PATH)
        if os.path.exists(partner_excel_abs):
            df_jscc = pd.read_excel(partner_excel_abs)
            df_jscc = df_jscc.dropna(subset=['SNR', 'MS-SSIM-jscc'])
            snr_jscc  = df_jscc['SNR'].values
            ssim_jscc = df_jscc['MS-SSIM-jscc'].values

            ax.plot(snr_jscc, ssim_jscc,
                    color='black', linestyle='--', linewidth=2.0, zorder=3,
                    label='JSCC')
            print("[叠绘成功] 已叠加合作方 JSCC 数据。")
        else:
            print(f"[警告] 未找到合作方数据文件: {partner_excel_abs}")
    except Exception as e:
        print(f"[警告] 解析合作方 JSCC 数据失败: {e}")

# ─────────────────────────────────────────────────────────────
# 步骤 4：图表元数据与保存路径
# ─────────────────────────────────────────────────────────────
ax.set_xlabel('SNR (dB)', fontsize=14)
ax.set_ylabel('MS-SSIM', fontsize=14)
# 修改坐标轴刻度数字的大小（建议设为 12）
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_title('JPEG Performance @ Different LDPC rates', fontsize=16)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.35)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(global_min_snr - 0.5, global_max_snr + 0.5)

# 独立前缀，避免覆盖原有图表
if ONLY_ORIGINAL_CLIFF:
    out_name = f'jpeg_envelope_{timestamp}.png'
else:
    out_name = f'jpeg_envelope_with_JSCC_{timestamp}.png'

out_path = os.path.join(WORKER_DIR, out_name)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n[出图完成] 包络线图已保存至: {out_name}")
