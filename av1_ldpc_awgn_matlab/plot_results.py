"""
plot_results.py  —  JSCC 悬崖效应极速可视化看板
从最新的 sim_av1_data_*.pkl 存档中读取仿真数据并生成图表。

使用方法：
    python plot_results.py

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
# 【开关】True: 仅绘制纯净版 Cliff 图; False: 叠加合作方的 JSCC Baseline 对比图
ONLY_ORIGINAL_CLIFF = True

# 【路径配置】
# 注意：如果是 AV1 的脚本，通配符为 'sim_av1_data_*.pkl'；如果是 JPEG 脚本，必须改为 'sim_jpeg_data_*.pkl'
PKL_PATTERN        = 'sim_av1_data_20260430_104509.pkl'
PARTNER_EXCEL_PATH = '../NTSCC-fixCBR-SNR-SSIM.xlsx'  # Excel 文件放在 Semantic_Debug 根目录，请根据实际情况调整相对路径

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
        "未找到任何仿真存档文件 (sim_av1_data_*.pkl)。\n"
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

    # 图例标签：底板极限码率特殊标注
    if is_floor:
        label = (f"Limit R≈{frac_str}  "
                 f"bpp={d['bpp']:.4f}  MS-SSIM={y_max:.4f}")
    else:
        label = f"R={frac_str}  bpp={d['bpp']:.4f}  MS-SSIM={y_max:.4f}"

    # 使用 plot 绘制真实斜率折线，不使用 step，不使用 marker='.'
    ax.plot(snr_arr, ssim_arr,
            color=colors[i], linewidth=2.0,
            label=label)

    # 寻找悬崖突破点：ssim_arr > 0 的第一个点
    cliff_indices = np.where(ssim_arr > 0)[0]
    if len(cliff_indices) > 0:
        cliff_idx = cliff_indices[0]
        cliff_snr = snr_arr[cliff_idx]
        ax.text(cliff_snr + 0.1, 0.05 + i * 0.04,
                f"Cliff={cliff_snr:.1f}dB",
                color=colors[i], fontsize=7, va='bottom')

# ─────────────────────────────────────────────────────────────
# 附加项：合作方 JSCC Baseline 叠绘
# ─────────────────────────────────────────────────────────────
if not ONLY_ORIGINAL_CLIFF:
    try:
        partner_excel_abs = os.path.join(WORKER_DIR, PARTNER_EXCEL_PATH)
        if os.path.exists(partner_excel_abs):
            # 自动寻找 openpyxl 引擎来读取真正的 Excel 文件
            df_jscc = pd.read_excel(partner_excel_abs)
            # 清洗数据，提取列
            df_jscc = df_jscc.dropna(subset=['SNR', 'MS-SSIM-jscc'])
            snr_jscc = df_jscc['SNR'].values
            ssim_jscc = df_jscc['MS-SSIM-jscc'].values

            # 绘制 Baseline，使用黑色虚线明显区分
            ax.plot(snr_jscc, ssim_jscc,
                    color='black', linestyle='--', linewidth=2.0, zorder=3,
                    label='JSCC')
            print("[叠绘成功] 已叠加合作方 JSCC 数据。")
        else:
            print(f"[警告] 未找到合作方数据文件: {partner_excel_abs}")
    except Exception as e:
        print(f"[警告] 解析合作方 JSCC 数据失败: {e}")

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('MS-SSIM', fontsize=12)
ax.set_title('MS-SSIM vs SNR (AV1/AVIF + 5G NR LDPC)', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.35)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(global_min_snr - 0.5, global_max_snr + 0.5)

# 根据开关状态和时间戳动态生成文件名
prefix = 'cliff_effect_av1'
if ONLY_ORIGINAL_CLIFF:
    out_name = f'{prefix}_{timestamp}.png'
else:
    out_name = f'{prefix}_with_JSCC_{timestamp}.png'

out_path = os.path.join(WORKER_DIR, out_name)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n[出图完成] 悬崖效应图已保存至: {out_name}")
