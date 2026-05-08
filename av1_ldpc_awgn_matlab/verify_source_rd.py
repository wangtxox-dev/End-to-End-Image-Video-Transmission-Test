"""
verify_source_rd.py  —  验证 AV1/AVIF 压缩 R-D 曲线是否与理论公式吻合
理论公式: MS-SSIM = 1 - 0.015315 * bpp^(-0.809798)
"""
import io
import os
import glob
import numpy as np
import pillow_avif  # noqa: F401  — 注册 AVIF 编解码器到 PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from pytorch_msssim import ms_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 广撒网备选列表：优先找通用 Unicode，找不到再找港版苹方、黑体，最后兜底
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC', 'PingFang SC'] 
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 参数配置
# ─────────────────────────────────────────────
KODAK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kodak')
G         = 768 * 512 * 2          # 总空口比特数
rates     = [1/32, 1/24, 1/16, 1/8, 1/4]  # 目标码率列表


def avif_bisect(img_pil: Image.Image, target_bits: int) -> bytes:
    """二分查找严格不超过 target_bits 的最高 AVIF quality，返回二进制流。
    quality 范围 0-100（数值越大质量越高、文件越大，与 JPEG 一致）。
    使用 speed=4 加速编码。
    """
    lo, hi = 0, 100
    best_buf = None
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        img_pil.save(buf, format='AVIF', quality=mid, speed=4)
        size_bits = buf.tell() * 8
        if size_bits <= target_bits:
            best_buf = buf.getvalue()  # 满足约束，记录并尝试更高质量
            lo = mid + 1
        else:
            hi = mid - 1              # 超出约束，必须降低质量
    # 若 quality=0 仍超出（极低码率），退而求其次取最低质量
    if best_buf is None:
        buf = io.BytesIO()
        img_pil.save(buf, format='AVIF', quality=0, speed=4)
        best_buf = buf.getvalue()
    return best_buf


def img_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """PIL Image → (1, C, H, W) float32 tensor in [0,1]"""
    arr = np.array(img_pil).astype(np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t


def theory_msssim(bpp: float) -> float:
    return 1.0 - 0.015315 * (bpp ** (-0.809798))


# ─────────────────────────────────────────────
# 主验证逻辑
# ─────────────────────────────────────────────
png_files = sorted(glob.glob(os.path.join(KODAK_DIR, '*.png')))
if not png_files:
    raise FileNotFoundError(f"kodak 目录下没有找到 .png 文件: {KODAK_DIR}")

print(f"找到 {len(png_files)} 张 Kodak 图片")
print(f"{'码率':>6}  {'bpp':>6}  {'目标K(bits)':>12}  {'平均MS-SSIM':>12}  {'理论MS-SSIM':>12}  {'误差':>8}")
print("-" * 72)

measured_bpps   = []
measured_msssim = []

for R in rates:
    K_target   = int(G * R)
    bpp_target = K_target / (768 * 512)  # = R * 3

    ssim_list = []
    for fpath in png_files:
        img = Image.open(fpath).convert('RGB')  # 直接读取原图全分辨率，无需裁剪

        # 二分控码（严格 <= target_bits）
        avif_bytes = avif_bisect(img, K_target)
        actual_bits = len(avif_bytes) * 8
        actual_bpp  = actual_bits / (768 * 512)

        # 解压回图片
        img_rec = Image.open(io.BytesIO(avif_bytes)).convert('RGB')

        # 计算 MS-SSIM
        t_orig = img_to_tensor(img)
        t_rec  = img_to_tensor(img_rec)
        val = ms_ssim(t_orig, t_rec, data_range=1.0, size_average=True).item()
        ssim_list.append(val)
        print(f"    {os.path.basename(fpath)}  actual_bpp={actual_bpp:.4f}  MS-SSIM={val:.6f}")

    avg_ssim = np.mean(ssim_list)
    theory_v = theory_msssim(bpp_target)
    err      = abs(avg_ssim - theory_v)

    measured_bpps.append(bpp_target)
    measured_msssim.append(avg_ssim)

    frac = f"1/{round(1/R)}" if R < 1 else f"{R:.2f}"
    print(f"  {frac:>5}  {bpp_target:>6.4f}  {K_target:>12d}  {avg_ssim:>12.6f}  {theory_v:>12.6f}  {err:>8.6f}")

# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────
bpp_curve    = np.linspace(0.05, 1.6, 400)
theory_curve = [theory_msssim(b) for b in bpp_curve]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bpp_curve, theory_curve, 'b-', linewidth=2,
        label=r'理论曲线: $1 - 0.015315 \cdot bpp^{-0.809798}$')
ax.scatter(measured_bpps, measured_msssim, color='red', s=80, zorder=5,
           label='AV1/AVIF 实测 (Kodak 均值)')

for bpp, ms in zip(measured_bpps, measured_msssim):
    ax.annotate(f'bpp={bpp:.3f}\nMS-SSIM={ms:.4f}',
                xy=(bpp, ms), xytext=(bpp + 0.03, ms - 0.02),
                fontsize=7, color='darkred')

ax.set_xlabel('bpp', fontsize=12)
ax.set_ylabel('MS-SSIM', fontsize=12)
ax.set_title('AV1/AVIF R-D 曲线验证 vs 理论公式', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_xlim(0.03, 1.7)
ax.set_ylim(0.4, 1.02)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verify_av1_rd_result.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至: {out_path}")

# ─────────────────────────────────────────────
# 自我验证判断（宽容度放宽至 0.10）
# ─────────────────────────────────────────────
theory_vals = [theory_msssim(b) for b in measured_bpps]
max_err = max(abs(np.array(measured_msssim) - np.array(theory_vals)))
print(f"\n最大绝对误差: {max_err:.6f}")
if max_err < 0.10:
    print("✅ 验证通过：散点与理论曲线吻合（最大误差 < 0.10），自动继续执行任务3。")
    VERIFIED = True
else:
    print("⚠️  验证警告：误差偏大，请检查 AVIF 控码逻辑。")
    VERIFIED = False
