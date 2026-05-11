%% verify_snr.m
% =========================================================================
% 物理层 SNR 能量严格对账探针
% 目标：验证 5G NR OFDM 时域加噪的 Nfft 补偿系数是否能在频域
%       完美还原设定的底噪方差。
%
% 测试流程：
%   1. 初始化 OFDM（25 RB, 30kHz SCS, 300 数据子载波, 14 符号）
%   2. 生成功率归一化的 QPSK 频域网格
%   3. 频转时（nrOFDMModulate）
%   4. 探针1：验证时域信号功率 ≈ 300/Nfft
%   5. 设定 SNR=5dB，计算目标频域噪声方差
%   6. 按 N0_time = noiseVar/Nfft 在时域精确加噪
%   7. 时转频（nrOFDMDemodulate）解调纯噪声
%   8. 探针2：测量频域噪声方差
%   9. 终极裁决：对比目标值与实测值
% =========================================================================

clc; clear; close all;

fprintf('=================================================================\n');
fprintf('  物理层 SNR 能量对账探针 — 5G NR OFDM Nfft 补偿系数验证\n');
fprintf('=================================================================\n\n');

%% ── 步骤 1：初始化 OFDM 配置 ─────────────────────────────────────────
fprintf('【步骤 1】初始化 5G NR OFDM 配置...\n');

carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30;   % 30 kHz SCS (mu=1)
carrier.NSizeGrid         = 25;   % 25 RB → 300 有效数据子载波/符号
carrier.NStartGrid        = 0;

numDataSC  = carrier.NSizeGrid * 12;   % = 300 个有效数据子载波
symPerSlot = 14;                        % 每 slot 14 个 OFDM 符号

ofdmInfo   = nrOFDMInfo(carrier);
Nfft       = double(ofdmInfo.Nfft);
sampleRate = ofdmInfo.SampleRate;

fprintf('  子载波间隔  : %d kHz\n', carrier.SubcarrierSpacing);
fprintf('  资源块数量  : %d RB\n',  carrier.NSizeGrid);
fprintf('  有效数据子载波 : %d\n',  numDataSC);
fprintf('  OFDM 符号数 : %d\n',    symPerSlot);
fprintf('  Nfft        : %d\n',    Nfft);
fprintf('  采样率      : %.3f MHz\n\n', sampleRate / 1e6);

%% ── 步骤 2：生成功率归一化的 QPSK 频域网格 ──────────────────────────
fprintf('【步骤 2】生成 QPSK 频域网格（理论功率 = 1.0）...\n');

rng(2024);   % 固定随机种子，保证可复现

% 生成 300×14 个随机 QPSK 符号
% QPSK 星座点：(±1 ± j) / sqrt(2)，每个符号功率严格 = 1.0
numSymbols = numDataSC * symPerSlot;   % 300 × 14 = 4200
bits_raw   = randi([0, 1], numSymbols * 2, 1);   % QPSK 每符号 2 bit
txSym_flat = nrSymbolModulate(bits_raw, 'QPSK');  % 归一化 QPSK，E[|s|²]=1

% 验证符号功率
sym_power = mean(abs(txSym_flat).^2);
fprintf('  QPSK 符号实测平均功率 : %.6f（理论值 = 1.0）\n', sym_power);

% 填充 300×14 资源网格
txGrid = reshape(txSym_flat, numDataSC, symPerSlot);

fprintf('  txGrid 尺寸 : %d × %d\n\n', size(txGrid, 1), size(txGrid, 2));

%% ── 步骤 3：频转时（OFDM 调制）─────────────────────────────────────
fprintf('【步骤 3】执行 nrOFDMModulate（频域 → 时域）...\n');

% nrOFDMModulate 内部：IFFT 后除以 sqrt(Nfft)
% 因此时域信号能量 = 频域能量 / Nfft
txWaveform = nrOFDMModulate(carrier, txGrid);

fprintf('  时域波形长度 : %d 个采样点\n\n', length(txWaveform));

%% ── 探针 1：验证时域信号功率 ─────────────────────────────────────────
fprintf('【探针 1】时域信号功率验证\n');
fprintf('  ─────────────────────────────────────────────────────────\n');

% 实测时域方差（复信号总功率 = 实部方差 + 虚部方差）
measured_signal_power = mean(abs(txWaveform).^2);

% 理论预测：只有 300 个子载波有能量，IFFT 归一化后
% 时域功率 = (有效子载波数 / Nfft) × 符号功率
%           = (300 / Nfft) × 1.0
theoretical_signal_power = numDataSC / Nfft;

ratio_signal = measured_signal_power / theoretical_signal_power;

fprintf('  实测时域信号功率   : %.8f\n', measured_signal_power);
fprintf('  理论时域信号功率   : %.8f  （= %d / %d）\n', ...
        theoretical_signal_power, numDataSC, Nfft);
fprintf('  实测 / 理论 比值   : %.6f\n', ratio_signal);

if abs(ratio_signal - 1.0) < 0.02
    fprintf('  ✓ 时域信号功率与理论值吻合（误差 < 2%%）\n');
else
    fprintf('  ✗ 时域信号功率偏差 %.2f%%，请检查 OFDM 配置\n', ...
            (ratio_signal - 1.0) * 100);
end
fprintf('  ─────────────────────────────────────────────────────────\n\n');

%% ── 步骤 5：设定发端底噪基准 ─────────────────────────────────────────
fprintf('【步骤 5】设定 SNR 基准与目标频域噪声方差...\n');

snr_dB     = 5;                          % 目标 SNR = 5 dB
snr_linear = 10^(snr_dB / 10);          % 线性 SNR

% 频域噪声方差：基于发端 QPSK 符号功率 = 1.0 定义
% noiseVar = E[|s|²] / SNR = 1.0 / snr_linear
target_noiseVar_freq = 1.0 / snr_linear;

fprintf('  SNR 设定值         : %d dB（线性 = %.4f）\n', snr_dB, snr_linear);
fprintf('  目标频域噪声方差   : %.8f\n\n', target_noiseVar_freq);

%% ── 步骤 6：时域精确加噪 ─────────────────────────────────────────────
fprintf('【步骤 6】时域精确加噪（N0_time = noiseVar / Nfft）...\n');

% 核心补偿逻辑：
%   nrOFDMModulate 执行 IFFT/sqrt(Nfft)，时域信号功率缩小 Nfft 倍。
%   nrOFDMDemodulate 执行 FFT/sqrt(Nfft)，频域恢复时放大 Nfft 倍。
%   因此，若要频域噪声方差 = noiseVar，
%   时域噪声方差必须 = noiseVar / Nfft。
N0_time = target_noiseVar_freq / Nfft;   % 时域复高斯噪声总方差

fprintf('  Nfft 补偿系数      : %d\n', Nfft);
fprintf('  时域噪声方差 N0_time : %.10f  （= %.8f / %d）\n', ...
        N0_time, target_noiseVar_freq, Nfft);

% 生成复高斯白噪声：实部和虚部各为 N(0, N0_time/2)
% 复信号总方差 = N0_time/2 + N0_time/2 = N0_time
noise_time = (randn(size(txWaveform)) + ...
              1j * randn(size(txWaveform))) * sqrt(N0_time / 2);

% 验证生成噪声的实际时域方差
measured_N0_time = mean(abs(noise_time).^2);
fprintf('  实测时域噪声方差   : %.10f（理论 = %.10f）\n', ...
        measured_N0_time, N0_time);
fprintf('  时域噪声方差误差   : %.4f%%\n\n', ...
        abs(measured_N0_time - N0_time) / N0_time * 100);

%% ── 步骤 7：时转频（解调纯噪声）─────────────────────────────────────
fprintf('【步骤 7】执行 nrOFDMDemodulate（纯噪声时域 → 频域）...\n');
fprintf('  注：直接解调纯噪声，排除信号干扰，最干净地测量噪声功率\n');

% 极其重要：直接解调纯噪声波形，不加信号
% nrOFDMDemodulate 内部执行 FFT/sqrt(Nfft)，将时域噪声放大 Nfft 倍
noiseGrid = nrOFDMDemodulate(carrier, noise_time);

fprintf('  noiseGrid 尺寸 : %d × %d\n\n', size(noiseGrid, 1), size(noiseGrid, 2));

%% ── 探针 2：频域底噪对账 ─────────────────────────────────────────────
fprintf('【探针 2】频域底噪方差对账\n');
fprintf('  ─────────────────────────────────────────────────────────\n');

% 截取有效数据区域：300 个数据子载波 × 14 个符号
% noiseGrid 的行数 = numDataSC = 300，列数 = symPerSlot = 14
noiseGrid_data = noiseGrid(1:numDataSC, 1:symPerSlot);

% 计算有效区域内噪声的真实方差
% 复信号方差 = E[|n|²] = E[Re(n)²] + E[Im(n)²]
noise_flat           = noiseGrid_data(:);
measured_noiseVar_freq = mean(abs(noise_flat).^2);

fprintf('  有效数据区域尺寸   : %d × %d = %d 个样本\n', ...
        numDataSC, symPerSlot, numDataSC * symPerSlot);
fprintf('  目标频域噪声方差   : %.8f\n', target_noiseVar_freq);
fprintf('  实测频域噪声方差   : %.8f\n', measured_noiseVar_freq);
fprintf('  ─────────────────────────────────────────────────────────\n\n');

%% ── 步骤 9：终极裁决 ─────────────────────────────────────────────────
fprintf('【终极裁决】Nfft 补偿系数验证结果\n');
fprintf('  ═════════════════════════════════════════════════════════\n');

deviation_ratio  = measured_noiseVar_freq / target_noiseVar_freq;
deviation_pct    = abs(deviation_ratio - 1.0) * 100;

fprintf('  目标频域噪声方差   : %.8f\n', target_noiseVar_freq);
fprintf('  实测频域噪声方差   : %.8f\n', measured_noiseVar_freq);
fprintf('  实测 / 目标 比值   : %.6f\n', deviation_ratio);
fprintf('  相对误差           : %.4f%%\n', deviation_pct);
fprintf('  ─────────────────────────────────────────────────────────\n');

if deviation_pct < 1.0
    fprintf('  ✅ 【对账成功】时域 Nfft 补偿系数完全正确！\n');
    fprintf('     N0_time = noiseVar / Nfft 的推导在物理上严格成立。\n');
    fprintf('     频域噪声方差与目标值误差 < 1%%，能量守恒验证通过。\n');
else
    fprintf('  ❌ 【对账失败】检测到明显偏差！\n');
    fprintf('     实测值是目标值的 %.4f 倍（偏差 %.2f%%）\n', ...
            deviation_ratio, deviation_pct);
    if deviation_ratio > 1.0
        fprintf('     → 频域噪声偏高：时域噪声方差设置过大，\n');
        fprintf('       或 Nfft 补偿系数应为 %.1f 而非 %d\n', ...
                Nfft * deviation_ratio, Nfft);
    else
        fprintf('     → 频域噪声偏低：时域噪声方差设置过小，\n');
        fprintf('       或 Nfft 补偿系数应为 %.1f 而非 %d\n', ...
                Nfft * deviation_ratio, Nfft);
    end
end

fprintf('  ═════════════════════════════════════════════════════════\n\n');

%% ── 附加信息：完整能量链路追踪 ──────────────────────────────────────
fprintf('【附录】完整能量链路追踪摘要\n');
fprintf('  ─────────────────────────────────────────────────────────\n');
fprintf('  频域符号功率（发端）  : %.6f\n', sym_power);
fprintf('  时域信号功率（实测）  : %.8f\n', measured_signal_power);
fprintf('  时域信号功率（理论）  : %.8f  = %d/%d\n', ...
        theoretical_signal_power, numDataSC, Nfft);
fprintf('  ─────────────────────────────────────────────────────────\n');
fprintf('  SNR 设定              : %d dB = %.4f（线性）\n', snr_dB, snr_linear);
fprintf('  目标频域噪声方差      : %.8f  = 1/%d\n', ...
        target_noiseVar_freq, round(snr_linear));
fprintf('  时域噪声方差 N0_time  : %.10f = noiseVar/%d\n', N0_time, Nfft);
fprintf('  实测时域噪声方差      : %.10f\n', measured_N0_time);
fprintf('  ─────────────────────────────────────────────────────────\n');
fprintf('  实测频域噪声方差      : %.8f\n', measured_noiseVar_freq);
fprintf('  目标频域噪声方差      : %.8f\n', target_noiseVar_freq);
fprintf('  最终误差              : %.4f%%\n', deviation_pct);
fprintf('  ─────────────────────────────────────────────────────────\n');
fprintf('  验证结论：N0_time = noiseVar / Nfft 补偿系数 %s\n', ...
        ternary_str(deviation_pct < 1.0, '✅ 正确', '❌ 有误'));
fprintf('  ═════════════════════════════════════════════════════════\n\n');

%% ── 辅助函数 ─────────────────────────────────────────────────────────
function s = ternary_str(cond, a, b)
    if cond
        s = a;
    else
        s = b;
    end
end
