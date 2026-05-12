"""
run_sim.py  —  JSCC 悬崖效应 (Cliff Effect) 仿真引擎 [OFDM 版]
架构：Python 负责信源编解码与评估，MATLAB 负责 5G NR OFDM 物理层传输
信源编码格式：AV1 / AVIF
信道模型：由 --channel / --ds 参数决定（AWGN 或 TDL-* 多径衰落）
信道估计：完美信道估计 (Perfect CSI) + MMSE 均衡

运行完毕后，仿真数据将以 pickle 格式落盘，供 plot_results.py 读取出图。

【扰码架构说明】
  扰码在 Python 侧实施（tx_bits 进入 MATLAB 前加扰，rx_bits 返回后解扰）。
  原因：MATLAB 链路为标准 5G NR 物理层（nrCRCEncode→nrLDPCEncode），
  在其内部插入扰码会破坏 CRC 语义完整性；Python 侧加扰对物理层完全透明，
  且扰码序列由固定种子 PRNG 生成，收发两端绝对一致，符合 3GPP 数据扰码规范。

用法示例：
    venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py
    venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py --channel TDL-C --ds 100e-9
    venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py --channel AWGN
    venv.nosync/bin/python av1_ldpc_ofdm_matlab/run_sim.py --image kodim08.png
"""

# matlab.engine.shareEngine('JSCC_Engine')

import argparse
import io
import os
import sys
import glob
import json
import pickle
import datetime
import numpy as np
from fractions import Fraction
import pillow_avif  # noqa: F401  — 注册 AVIF 编解码器到 PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from pytorch_msssim import ms_ssim

# ── 命令行参数解析 ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="JSCC 悬崖效应仿真引擎 [OFDM 版]")
parser.add_argument("--channel", type=str, default="TDL-C",
                    help="信道模型，如 AWGN、TDL-C、TDL-D")
parser.add_argument("--ds", type=float, default=300e-9,
                    help="时延扩展（秒），AWGN 时忽略")
parser.add_argument("--image", type=str, default="kodim01.png",
                    help="指定测试图片名，如 kodim08.png")
args = parser.parse_args()

CHANNEL_MODEL = args.channel
DELAY_SPREAD  = args.ds

# ─────────────────────────────────────────────────────────────
# 预加载信道容量曲线（由 build_capacity_curve.py 生成）
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 根据信道参数动态确定 JSON 文件名
if CHANNEL_MODEL == "AWGN":
    _CURVE_JSON = os.path.join(SCRIPT_DIR, "capacity_curve_AWGN.json")
else:
    ds_ns = int(round(DELAY_SPREAD * 1e9))
    _CURVE_JSON = os.path.join(SCRIPT_DIR, f"capacity_curve_{CHANNEL_MODEL}_{ds_ns}ns.json")

_YELLOW = "\033[33m"
_RESET  = "\033[0m"

_CURVE_LOADED = False
_CURVE_SNR_DB   = None
_CURVE_CAPACITY = None

if os.path.exists(_CURVE_JSON):
    with open(_CURVE_JSON, "r", encoding="utf-8") as _f:
        _curve_data = json.load(_f)
    _CURVE_SNR_DB   = np.array(_curve_data["snr_db"],   dtype=np.float64)
    _CURVE_CAPACITY = np.array(_curve_data["capacity"], dtype=np.float64)
    _CURVE_LOADED   = True
    print(f"[容量曲线] 已预加载 {len(_CURVE_SNR_DB)} 个数据点，"
          f"SNR 范围 [{_CURVE_SNR_DB[0]:.1f}, {_CURVE_SNR_DB[-1]:.1f}] dB。"
          f"  信道: {CHANNEL_MODEL}")
else:
    print(f"{_YELLOW}[警告] 找不到信道容量曲线缓存文件：{_CURVE_JSON}\n"
          f"       将使用 AWGN 香农极限作为扫频备用起点（物理保底，安全降级）。\n"
          f"       如需精确极限，请先运行：\n"
          f"         venv.nosync/bin/python av1_ldpc_ofdm_matlab/build_capacity_curve.py "
          f"--channel {CHANNEL_MODEL}"
          + (f" --ds {DELAY_SPREAD}" if CHANNEL_MODEL != "AWGN" else "")
          + f"{_RESET}")

# ─────────────────────────────────────────────────────────────
# 参数配置
# ─────────────────────────────────────────────────────────────
USE_IMAGE     = True                              # True: AVIF 真实压缩; False: 随机比特
G             = 768 * 512 * 2                     # 空口总比特数 (786432)，Kodak 全分辨率
RATES         = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2]          # 目标码率列表

# kodak 文件夹已移动到 Semantic_Debug 根目录，使用绝对路径定位
WORKER_DIR    = SCRIPT_DIR                              # sim_ofdm_worker.m 所在目录
KODAK_DIR     = os.path.abspath(os.path.join(WORKER_DIR, '../kodak'))
DEFAULT_IMAGE = 'kodim01.png'          # 默认测试图片名
AVIF_SPEED    = 4                      # AVIF 编码速度 (0-10，数值越大越快但压缩率略低)

IMAGE_TO_USE = args.image if args.image else DEFAULT_IMAGE

# ─────────────────────────────────────────────────────────────
# 扰码器（Scrambler）— Python 侧实施，对 MATLAB 物理层完全透明
# ─────────────────────────────────────────────────────────────
_SCRAMBLE_SEED = 42

def make_scramble_seq(length: int) -> np.ndarray:
    """生成长度为 length 的伪随机扰码序列（0/1 int32 数组）。"""
    rng = np.random.default_rng(_SCRAMBLE_SEED)
    return rng.integers(0, 2, size=length, dtype=np.int32)


def scramble(bits: np.ndarray) -> np.ndarray:
    """对比特流做 XOR 加扰，返回加扰后的 int32 数组。"""
    seq = make_scramble_seq(len(bits))
    return np.bitwise_xor(bits.astype(np.int32), seq)


def descramble(bits: np.ndarray) -> np.ndarray:
    """对比特流做 XOR 解扰（XOR 自逆，与加扰操作完全相同）。"""
    return scramble(bits)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def format_rate(r: float) -> str:
    """将浮点码率精确转为分数字符串。"""
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


def img_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """PIL Image → (1, C, H, W) float32 tensor in [0,1]"""
    arr = np.array(img_pil).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def calc_shannon_limit_dB(R: float, Q: int = 2) -> float:
    """AWGN 香农极限（保留备用）"""
    snr_linear = 2 ** (R * Q) - 1
    return 10.0 * np.log10(snr_linear)


def calc_channel_limit_dB(R: float) -> float:
    """利用预加载的容量曲线，通过一维线性插值求 SNR 极限（dB）。
    目标容量 target_cap = R * 2.0（QPSK 每符号 2 bit）。
    仅在 _CURVE_LOADED=True 时可调用，否则返回 None。
    """
    if not _CURVE_LOADED:
        return None
    target_cap = R * 2.0
    return float(np.interp(target_cap, _CURVE_CAPACITY, _CURVE_SNR_DB))


def avif_bisect(img_pil: Image.Image, target_bits: int) -> tuple[bytes, int]:
    """二分查找严格不超过 target_bits 的最高 AVIF quality。"""
    lo, hi = 0, 100
    best_buf = None
    best_quality = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        img_pil.save(buf, format='AVIF', quality=mid, speed=AVIF_SPEED)
        size_bits = buf.tell() * 8
        if size_bits <= target_bits:
            best_buf = buf.getvalue()
            best_quality = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best_buf is None:
        buf = io.BytesIO()
        img_pil.save(buf, format='AVIF', quality=0, speed=AVIF_SPEED)
        best_buf = buf.getvalue()
        best_quality = 0
    return best_buf, best_quality


def get_tx_bits(R: float) -> tuple[np.ndarray, object, int, int]:
    """根据 USE_IMAGE 开关获取信源比特流。"""
    K = int(G * R)
    if not USE_IMAGE:
        return np.random.randint(0, 2, K, dtype=np.int32), None, -1, K

    png_files = sorted(glob.glob(os.path.join(KODAK_DIR, IMAGE_TO_USE)))
    if not png_files:
        raise FileNotFoundError(f"kodak 目录下找不到指定图片: {KODAK_DIR}/{IMAGE_TO_USE}")

    img = Image.open(png_files[0]).convert('RGB')
    print(f"  [图片] 使用: {os.path.basename(png_files[0])}")
    avif_bytes, quality = avif_bisect(img, K)

    bits = np.unpackbits(np.frombuffer(avif_bytes, dtype=np.uint8))
    actual_source_len = len(bits)

    if actual_source_len < K:
        bits = np.pad(bits, (0, K - actual_source_len), mode='constant')

    bits = bits.astype(np.int32)
    return bits, img, quality, actual_source_len


# ─────────────────────────────────────────────────────────────
# 启动 MATLAB Engine
# ─────────────────────────────────────────────────────────────
print("正在连接后台 MATLAB 共享引擎...")
import matlab.engine
eng = matlab.engine.connect_matlab('JSCC_Engine')
eng.addpath(WORKER_DIR, nargout=0)
print("MATLAB Engine 启动成功。\n")

# ─────────────────────────────────────────────────────────────
# 任务1：前置探针 — 探测 AVIF 信源物理底线
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("【前置探针】正在探测 AVIF 信源物理底线...")

_png_files = sorted(glob.glob(os.path.join(KODAK_DIR, IMAGE_TO_USE)))
if not _png_files:
    raise FileNotFoundError(f"kodak 目录下找不到指定图片: {KODAK_DIR}")
print(f"  [探针] 目标图片: {os.path.basename(_png_files[0])}")
_probe_img = Image.open(_png_files[0]).convert('RGB')

_buf_min = io.BytesIO()
_probe_img.save(_buf_min, format='AVIF', quality=0, speed=AVIF_SPEED)
K_min = _buf_min.tell() * 8
R_min = K_min / G

print(f"  AVIF quality=0,speed={AVIF_SPEED} 最小压缩: K_min={K_min} bits, R_min={R_min:.6f} (≈1/{round(1/R_min)})")
print("=" * 60)

ORANGE = "\033[33m"
RESET  = "\033[0m"

valid_rates = []
for R in RATES:
    if R >= R_min:
        valid_rates.append(R)
    else:
        frac_orig = format_rate(R)
        frac_min  = format_rate(R_min)
        print(f"{ORANGE}[底板触发] 目标码率 R={frac_orig} 低于物理极限，"
              f"强制校准为实际极限 R_min≈{frac_min} (K={K_min}){RESET}")
        valid_rates.append(R_min)

valid_rates = sorted(list(set(valid_rates)), reverse=True)
print(f"有效码率列表（去重后）: {[format_rate(r) for r in valid_rates]}\n")

# ─────────────────────────────────────────────────────────────
# 主循环：遍历有效码率，动态 SNR 扫频
# ─────────────────────────────────────────────────────────────
results = {}

for R in valid_rates:
    is_floor = (R == R_min) and (R_min not in RATES)

    K   = int(G * R)
    bpp = K / (768 * 512)

    # ── 信道容量极限（通过预加载曲线插值，或降级为 AWGN 保底）──
    snr_lim_awgn = calc_shannon_limit_dB(R, Q=2)   # AWGN 香农极限，物理绝对下限

    if _CURVE_LOADED:
        print(f"  [信道容量] 正在查表插值码率 R={format_rate(R)} 的容量极限...")
        snr_lim     = calc_channel_limit_dB(R)      # float，当前信道真实极限
        snr_lim_str = f"{snr_lim:.2f}"              # 安全字符串，用于 print
        scan_start  = snr_lim                       # 从真实极限起步
    else:
        snr_lim     = None                          # 无缓存，极限未知
        snr_lim_str = "N/A"                         # 安全字符串，用于 print
        scan_start  = snr_lim_awgn                  # 降级：AWGN 极限作为物理保底起点

    # ── 动态扫频：统一从 scan_start 起步，覆盖 6 dB 窗口 ──────
    snr_array = np.arange(scan_start - 5, scan_start + 6, 0.5)

    frac_str = format_rate(R)
    print(f"{'='*60}")
    print(f"码率 R={frac_str}  bpp={bpp:.4f}  K={K}  "
          f"[当前信道容量极限]={snr_lim_str} dB  (AWGN参考={snr_lim_awgn:.2f} dB)"
          + ("  [底板极限]" if is_floor else ""))
    print(f"SNR 扫频: [{snr_array[0]:.1f}, {snr_array[-1]:.1f}] dB  步进 0.5 dB")
    print(f"{'='*60}")

    tx_bits, ref_img, quality, actual_source_len = get_tx_bits(R)
    print(f"  [信源] AV1 Quality={quality}  K={K}  压缩后比特数={actual_source_len}")

    tx_bits_scrambled = scramble(tx_bits)
    t_ref = img_to_tensor(ref_img) if ref_img is not None else None
    tx_bits_ml = matlab.int32(tx_bits_scrambled.tolist())

    snr_list        = []
    ssim_list       = []
    cliff_reported  = False
    saved_this_rate = False
    y_max           = 0.0

    for snr in snr_array:
        # ── 调用 OFDM 物理层 worker（补齐 channelModel 和 delaySpread）──
        ber, rx_bits_ml, H_energy_ml = eng.sim_ofdm_worker(
            tx_bits_ml, float(R), float(G), float(snr),
            CHANNEL_MODEL, float(DELAY_SPREAD),
            nargout=3
        )
        ber      = float(ber)
        H_energy = float(H_energy_ml)                          # 本次快照真实平均信道能量
        # 等效物理 SNR = 标称 SNR + 10*log10(H_energy)
        # H_energy > 1 → 信道"赏饭"，等效 SNR 高于标称；< 1 → 信道"扣饭"
        H_energy_dB  = 10.0 * np.log10(H_energy) if H_energy > 0 else float('-inf')
        equiv_snr_dB = snr + H_energy_dB

        if ber > 0:
            y = 0.0
        else:
            rx_bits_np = np.array(rx_bits_ml, dtype=np.int32).flatten()
            rx_bits_np = descramble(rx_bits_np)

            rx_bits_u8 = rx_bits_np.astype(np.uint8)
            n_bits     = (len(rx_bits_u8) // 8) * 8
            rx_bits_u8 = rx_bits_u8[:n_bits]
            rx_bytes   = np.packbits(rx_bits_u8).tobytes()

            try:
                img_rec = Image.open(io.BytesIO(rx_bytes)).convert('RGB')
                if t_ref is not None:
                    t_rec = img_to_tensor(img_rec)
                    y = ms_ssim(t_ref, t_rec, data_range=1.0, size_average=True).item()
                else:
                    y = 1.0

                if y > y_max:
                    y_max = y

                if not saved_this_rate and ref_img is not None:
                    rate_tag  = f"R{format_rate_file(R)}"
                    snr_tag   = f"{snr:.1f}"
                    save_name = f"recovered_ofdm_{rate_tag}_SNR_{snr_tag}.png"
                    save_path = os.path.join(WORKER_DIR, save_name)
                    img_rec.save(save_path)
                    print(f"  [存图] 恢复图已保存: {save_name}  Quality={quality}")
                    saved_this_rate = True

            except Exception as e:
                print(f"  [警告] 图像解码失败 (SNR={snr:.1f} dB): {e}")
                y = 0.0

            if not cliff_reported and y > 0:
                print(f"  [报捷] 码率={frac_str} | bpp={bpp:.4f} | "
                      f"[当前信道容量极限]={snr_lim_str} dB | "
                      f"AWGN参考={snr_lim_awgn:.2f} dB | "
                      f"临界SNR={snr:.1f} dB | "
                      f"Quality={quality} | "
                      f"实测 MS-SSIM={y:.6f}")
                print(f"         ↳ 实测信道能量 H_energy={H_energy:.6f} ({H_energy_dB:+.3f} dB) | "
                      f"等效物理SNR={equiv_snr_dB:.2f} dB ")
                cliff_reported = True

        snr_list.append(snr)
        ssim_list.append(y)
        print(f"  SNR={snr:+6.1f} dB | BER={ber:.2e} | MS-SSIM={y:.4f} | "
              f"H_energy={H_energy:.4f} ({H_energy_dB:+.2f} dB) | "
              f"Equiv_SNR={equiv_snr_dB:.2f} dB")

    results[R] = {'snr_list': snr_list, 'ssim_list': ssim_list,
                  'snr_lim': snr_lim,           # float 或 None（降级时）
                  'snr_lim_awgn': snr_lim_awgn,
                  'bpp': bpp, 'y_max': y_max,
                  'is_floor': is_floor, 'quality': quality}

# eng.quit()  # 绝对不要 quit，留着下次秒连！
print("\n仿真结束，MATLAB Engine 仍在后台待命。")

# ─────────────────────────────────────────────────────────────
# 数据落盘
# ─────────────────────────────────────────────────────────────
timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
pkl_filename  = f"sim_av1_ofdm_data_{timestamp_str}.pkl"
pkl_path      = os.path.join(WORKER_DIR, pkl_filename)

payload = {
    'valid_rates':    valid_rates,
    'results':        results,
    'G':              G,
    'timestamp':      timestamp_str,
    'channel_model':  CHANNEL_MODEL,
    'delay_spread':   DELAY_SPREAD,
}

with open(pkl_path, 'wb') as f:
    pickle.dump(payload, f)

print(f"[数据落盘] 仿真数据已保存至: {pkl_filename}")
