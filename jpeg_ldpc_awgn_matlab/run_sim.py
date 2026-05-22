"""
run_sim.py  —  JSCC 悬崖效应 (Cliff Effect) 仿真引擎
架构：Python 负责信源编解码与评估，MATLAB 负责 5G NR 物理层传输
信源编码格式：JPEG

运行完毕后，仿真数据将以 pickle 格式落盘，供 plot_results.py 读取出图。

【扰码架构说明】
  扰码在 Python 侧实施（tx_bits 进入 MATLAB 前加扰，rx_bits 返回后解扰）。
  原因：MATLAB 链路为标准 5G NR 物理层（nrCRCEncode→nrLDPCEncode），
  在其内部插入扰码会破坏 CRC 语义完整性；Python 侧加扰对物理层完全透明，
  且扰码序列由固定种子 PRNG 生成，收发两端绝对一致，符合 3GPP 数据扰码规范。
"""

# matlab.engine.shareEngine('JSCC_Engine')

import io
import os
import sys
import glob
import pickle
import datetime
import numpy as np
from fractions import Fraction
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from pytorch_msssim import ms_ssim

# ─────────────────────────────────────────────────────────────
# 参数配置
# ─────────────────────────────────────────────────────────────
DEFAULT_IMAGE = 'kodim01.png'          # 默认测试图片名 (当命令行不指定时使用)
USE_IMAGE  = True                          # True: JPEG真实压缩; False: 随机比特
G          = round(768 * 512 * 0.5)    # 空口总比特数 (1179648)，Kodak 全分辨率
RATES      = [1/4, 1/3, 1/2, 2/3, 4/5]  # 目标码率列表
KODAK_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kodak')
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))  # sim_awgn_worker.m 所在目录

# ── 命令行参数：支持 --image kodim01.png 指定测试图片 ──────────
# 用法: python run_sim.py --image kodim01.png
_TARGET_IMAGE = DEFAULT_IMAGE
for _i, _arg in enumerate(sys.argv[1:]):
    if _arg == '--image' and _i + 1 < len(sys.argv) - 1:
        _TARGET_IMAGE = sys.argv[_i + 2]
        break

# ─────────────────────────────────────────────────────────────
# 扰码器（Scrambler）— Python 侧实施，对 MATLAB 物理层完全透明
# ─────────────────────────────────────────────────────────────
# 扰码种子固定为 42，保证收发两端序列绝对一致。
# 使用独立的 np.random.Generator 实例，不污染全局随机状态。
_SCRAMBLE_SEED = 42

def make_scramble_seq(length: int) -> np.ndarray:
    """生成长度为 length 的伪随机扰码序列（0/1 int32 数组）。
    每次调用使用相同种子，保证收发一致性。
    """
    rng = np.random.default_rng(_SCRAMBLE_SEED)
    return rng.integers(0, 2, size=length, dtype=np.int32)


def scramble(bits: np.ndarray) -> np.ndarray:
    """对比特流做 XOR 加扰，返回加扰后的 int32 数组。"""
    seq = make_scramble_seq(len(bits))
    return np.bitwise_xor(bits.astype(np.int32), seq)


def descramble(bits: np.ndarray) -> np.ndarray:
    """对比特流做 XOR 解扰（XOR 自逆，与加扰操作完全相同）。"""
    return scramble(bits)  # XOR 自逆


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


def img_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """PIL Image → (1, C, H, W) float32 tensor in [0,1]，与 verify_source_rd.py 保持一致"""
    arr = np.array(img_pil).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t


def calc_shannon_limit_dB(R: float, Q: int = 2) -> float:
    """严格对齐 calc_shannon_limit.m: snr_limit_linear = 2^(R*Q) - 1"""
    snr_linear = 2 ** (R * Q) - 1
    return 10.0 * np.log10(snr_linear)


def jpeg_bisect(img_pil: Image.Image, target_bits: int) -> tuple:
    """二分查找严格不超过 target_bits 的最高 JPEG quality，返回 (二进制流, quality)。"""
    lo, hi = 1, 95
    best_buf = None
    best_quality = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=mid)
        size_bits = buf.tell() * 8
        if size_bits <= target_bits:
            best_buf = buf.getvalue()  # 满足约束，记录并尝试更高质量
            best_quality = mid
            lo = mid + 1
        else:
            hi = mid - 1              # 超出约束，必须降低质量
    return best_buf, best_quality


def get_tx_bits(R: float) -> tuple:
    """根据 USE_IMAGE 开关获取信源比特流，同时返回原始 PIL Image 作为 GT 参考。
    返回: (bits: np.ndarray shape=(K,) int32, ref_img: PIL.Image or None,
           quality: int, actual_source_len: int)

    生死防线：np.pad 补零机制绝对不可删除，确保与 AV1 分支物理层实验平面对称。
    """
    K = int(G * R)
    if not USE_IMAGE:
        return np.random.randint(0, 2, K, dtype=np.int32), None, -1, K

    png_files = sorted(glob.glob(os.path.join(KODAK_DIR, _TARGET_IMAGE)))
    if not png_files:
        raise FileNotFoundError(f"kodak 目录下没有 .png 文件: {KODAK_DIR}")

    # 取第一张图做代表（评估悬崖效应，单张即可），直接读取原图全分辨率
    img = Image.open(png_files[0]).convert('RGB')
    print(f"  [图片] 使用: {os.path.basename(png_files[0])}")
    jpeg_bytes, quality = jpeg_bisect(img, K)

    bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))
    actual_source_len = len(bits)  # 记录 np.pad 补零之前的真实 JPEG 压缩长度

    if actual_source_len < K:
        # 尾部补零填满 K —— 保持与 AV1 分支对称的物理层实验平面
        bits = np.pad(bits, (0, K - actual_source_len), mode='constant')
    elif actual_source_len > K:
        bits = bits[:K]  # 理论上新版 bisect 不会走到这里，作为最后防线

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
# 任务1：前置探针 — 探测 JPEG 信源物理底线
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("【前置探针】正在探测 JPEG 信源物理底线...")

_png_files = sorted(glob.glob(os.path.join(KODAK_DIR, _TARGET_IMAGE)))
if not _png_files:
    raise FileNotFoundError(f"kodak 目录下没有 .png 文件: {KODAK_DIR}")
print(f"  [探针] 目标图片: {os.path.basename(_png_files[0])}")
_probe_img = Image.open(_png_files[0]).convert('RGB')

# 使用 quality=1 进行无约束最低画质压缩，获取绝对最小字节数
_buf_min = io.BytesIO()
_probe_img.save(_buf_min, format='JPEG', quality=1)
K_min = _buf_min.tell() * 8          # 绝对最小比特数
R_min = K_min / G                    # 绝对物理极限码率

print(f"  JPEG quality=1 最小压缩: K_min={K_min} bits, R_min={R_min:.6f} (≈1/{round(1/R_min)})")
print("=" * 60)

# 洗牌与去重：构建有效码率列表
ORANGE = "\033[33m"  # 终端橙色/黄色
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

# 去重并按从大到小排序
valid_rates = sorted(list(set(valid_rates)), reverse=True)
print(f"有效码率列表（去重后）: {[format_rate(r) for r in valid_rates]}\n")

# ─────────────────────────────────────────────────────────────
# 主循环：遍历有效码率，动态 SNR 扫频
# ─────────────────────────────────────────────────────────────
results = {}   # {R: {'snr_list': [...], 'ssim_list': [...], 'is_floor': bool}}

for R in valid_rates:
    # 判断该码率是否为底板校准产生的极限码率
    is_floor = (R == R_min) and (R_min not in RATES)

    K       = int(G * R)
    bpp     = K / (768 * 512)   # = R * 3，基于 Kodak 全分辨率像素数
    snr_lim = calc_shannon_limit_dB(R, Q=2)

    # SNR 扫频范围：[极限-5, 极限+5]，步进 0.5 dB
    snr_array = np.arange(snr_lim - 5.0, snr_lim + 5.0 + 1e-9, 0.5)

    frac_str = format_rate(R)
    print(f"{'='*60}")
    print(f"码率 R={frac_str}  bpp={bpp:.4f}  K={K}  香农极限={snr_lim:.2f} dB"
          + ("  [底板极限]" if is_floor else ""))
    print(f"SNR 扫频: [{snr_array[0]:.1f}, {snr_array[-1]:.1f}] dB  步进 0.5 dB")
    print(f"{'='*60}")

    # 获取信源比特（每个码率只压缩一次，SNR 扫频复用同一份 tx_bits）
    tx_bits, ref_img, quality, actual_source_len = get_tx_bits(R)
    print(f"[信源] JPEG Quality={quality}  K={K}  压缩后比特数={actual_source_len}")

    # ── 加扰：在发送端对 tx_bits 做 XOR 扰码 ──────────────────
    # 目的：打散 padding 0 区域的全零序列，消除 LDPC 全零陷阱
    tx_bits_scrambled = scramble(tx_bits)

    # 预计算参考图张量（用于 MS-SSIM 计算）
    t_ref = img_to_tensor(ref_img) if ref_img is not None else None

    # 转为 MATLAB 可接受的类型（传入加扰后的比特）
    tx_bits_ml = matlab.int32(tx_bits_scrambled.tolist())

    snr_list        = []
    ssim_list       = []
    cliff_reported  = False
    saved_this_rate = False
    y_max           = 0.0   # 将由首次 ber==0 时的实测 MS-SSIM 填充

    for snr in snr_array:
        ber, rx_bits_ml = eng.sim_awgn_worker(tx_bits_ml, float(R), float(G), float(snr),
                                              nargout=2)
        ber = float(ber)

        if ber > 0:
            # JPEG 头可能已损坏，直接置 0，不尝试解码
            y = 0.0
        else:
            # ber == 0：比特完全正确，先解扰再尝试图像恢复
            rx_bits_np = np.array(rx_bits_ml, dtype=np.int32).flatten()

            # ── 解扰：还原原始信源比特 ────────────────────────────
            rx_bits_np = descramble(rx_bits_np)

            # 转为 uint8 并打包为字节流
            rx_bits_u8 = rx_bits_np.astype(np.uint8)
            # 确保比特长度是 8 的倍数
            n_bits     = (len(rx_bits_u8) // 8) * 8
            rx_bits_u8 = rx_bits_u8[:n_bits]
            rx_bytes   = np.packbits(rx_bits_u8).tobytes()

            try:
                img_rec = Image.open(io.BytesIO(rx_bytes)).convert('RGB')
                if t_ref is not None:
                    t_rec = img_to_tensor(img_rec)
                    y = ms_ssim(t_ref, t_rec, data_range=1.0, size_average=True).item()
                else:
                    y = 1.0  # 随机比特模式无参考图，直接置满分

                # 更新 y_max（取所有 ber==0 点中的最大值）
                if y > y_max:
                    y_max = y

                # 仅在首次达成 ber==0 时存图
                if not saved_this_rate and ref_img is not None:
                    rate_tag  = f"R{format_rate_file(R)}"
                    snr_tag   = f"{snr:.1f}"
                    save_name = f"recovered_{rate_tag}_SNR_{snr_tag}.png"
                    save_path = os.path.join(WORKER_DIR, save_name)
                    img_rec.save(save_path)
                    print(f"  [存图] 恢复图已保存: {save_name}  Quality={quality}")
                    saved_this_rate = True

            except Exception as e:
                print(f"  [警告] 图像解码失败 (SNR={snr:.1f} dB): {e}")
                y = 0.0

            if not cliff_reported and y > 0:
                print(f"  [报捷] 码率={frac_str} | bpp={bpp:.4f} | "
                      f"香农极限={snr_lim:.2f} dB | "
                      f"临界SNR={snr:.1f} dB | "
                      f"Quality={quality} | "
                      f"实测 MS-SSIM={y:.6f}")
                cliff_reported = True

        snr_list.append(snr)
        ssim_list.append(y)
        print(f"  SNR={snr:+6.1f} dB | BER={ber:.2e} | MS-SSIM={y:.4f}")

    results[R] = {'snr_list': snr_list, 'ssim_list': ssim_list,
                  'snr_lim': snr_lim, 'bpp': bpp, 'y_max': y_max,
                  'is_floor': is_floor, 'quality': quality}

# eng.quit()  # 绝对不要 quit，留着下次秒连！
print("\n仿真结束，MATLAB Engine 仍在后台待命。")

# ─────────────────────────────────────────────────────────────
# 数据落盘：序列化仿真结果为 pkl 存档
# ─────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
pkl_name  = f'sim_jpeg_data_{timestamp}.pkl'
pkl_path  = os.path.join(WORKER_DIR, pkl_name)

payload = {
    'valid_rates': valid_rates,
    'results':     results,
    'G':           G,
    'timestamp':   timestamp,
}

with open(pkl_path, 'wb') as f:
    pickle.dump(payload, f)

print(f"\n[数据落盘] 仿真存档已保存至: {pkl_name}")
print("[提示] 运行 plot_results.py 可直接读取存档并生成图表，无需重跑仿真。")
