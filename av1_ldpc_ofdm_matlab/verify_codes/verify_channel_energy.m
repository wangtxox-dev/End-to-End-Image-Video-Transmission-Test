%% verify_channel_energy.m
% =========================================================================
% TDL 信道跨频域能量守恒验证探针
%
% 物理假设：nrTDLChannel 将总能量（均值 1.0）归一化到整个系统采样带宽
%   （512 个 FFT 子载波）。由于分数时延滤波器的低通边缘衰减，
%   边缘保护间隔子载波能量被削弱，导致中心有效通带（300 个激活子载波）
%   获得额外能量补偿，平均功率被异常放大（预计约 1.45 倍）。
%
% Test 1：验证全频带（512 点）能量守恒 → 预期 ≈ 1.0
% Test 2：验证激活频带（300 点）能量被放大 → 预期 ≈ 1.45
% =========================================================================

clc; clear; close all;

fprintf('=================================================================\n');
fprintf('  TDL 信道跨频域能量守恒验证探针\n');
fprintf('=================================================================\n\n');

%% ── 基础配置 ─────────────────────────────────────────────────────────────
fprintf('【基础配置】初始化 5G NR OFDM 参数...\n');

carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 30;   % 30 kHz SCS (mu=1)
carrier.NSizeGrid         = 25;   % 25 RB → 300 有效数据子载波/符号
carrier.NStartGrid        = 0;

ofdmInfo   = nrOFDMInfo(carrier);
Nfft       = double(ofdmInfo.Nfft);
sampleRate = ofdmInfo.SampleRate;

fprintf('  子载波间隔     : %d kHz\n',    carrier.SubcarrierSpacing);
fprintf('  资源块数量     : %d RB\n',     carrier.NSizeGrid);
fprintf('  有效数据子载波 : %d\n',        carrier.NSizeGrid * 12);
fprintf('  Nfft           : %d\n',        Nfft);
fprintf('  采样率         : %.4f MHz\n\n', sampleRate / 1e6);

% 配置 TDL-D 信道（时延扩展 30ns，无多普勒）
channel = nrTDLChannel;
channel.DelayProfile        = 'TDL-D';
channel.DelaySpread         = 30e-9;
channel.MaximumDopplerShift = 0;
channel.SampleRate          = sampleRate;
channel.NumTransmitAntennas = 1;
channel.NumReceiveAntennas  = 1;

fprintf('  信道模型       : %s\n',        channel.DelayProfile);
fprintf('  时延扩展       : %.0f ns\n',   channel.DelaySpread * 1e9);
fprintf('  最大多普勒频移 : %d Hz\n\n',   channel.MaximumDopplerShift);

%% ── Test 1：验证全频带（512 点）能量守恒 ────────────────────────────────
fprintf('=================================================================\n');
fprintf('  Test 1: 全频带（512 点）能量守恒验证\n');
fprintf('=================================================================\n');
fprintf('  原理：对冲激响应做 512 点 FFT，得到完整系统频率响应 H(f)。\n');
fprintf('        若信道能量归一化到全带宽，则 mean(|H|²) 应 ≈ 1.0。\n\n');

% 生成时域理想冲激信号（长度 1024，足够捕获 TDL 全部多径）
tx_impulse = [1; zeros(1023, 1)];

% 重置信道状态，确保每次测试独立
reset(channel);
rx_impulse = channel(tx_impulse);

% 取接收信号的第一列（单天线输出）
rx_impulse = rx_impulse(:, 1);

% 512 点 FFT → 全频带频率响应
H_full = fft(rx_impulse, Nfft);

% 计算 512 个频点的平均功率
power_full = mean(abs(H_full).^2);

fprintf('  冲激信号长度   : %d 个采样点\n', length(tx_impulse));
fprintf('  FFT 点数       : %d\n',          Nfft);
fprintf('  H_full 长度    : %d\n',          length(H_full));
fprintf('\n');
fprintf('  >>> Test 1: 全频带 (512 点) 平均能量 = %.6f\n', power_full);
fprintf('      预期值 ≈ 1.0（信道总能量守恒）\n');
fprintf('      偏差   = %.4f%%\n\n', abs(power_full - 1.0) * 100);

%% ── Test 2：验证激活频带（300 点）能量被放大 ────────────────────────────
fprintf('=================================================================\n');
fprintf('  Test 2: 中心激活频带（300 点）能量放大验证\n');
fprintf('=================================================================\n');
fprintf('  原理：通过标准 OFDM 调制/解调流程，提取 300 个有效子载波的\n');
fprintf('        信道响应 H_active = rxGrid / txGrid。\n');
fprintf('        若边缘子载波能量被低通滤波器削弱，中心通带将获得补偿，\n');
fprintf('        mean(|H_active|²) 应显著 > 1.0（预期 ≈ 1.45）。\n\n');

numDataSC  = carrier.NSizeGrid * 12;   % 300
symPerSlot = 14;

% 生成全 1 参考网格（300 × 14 × 1）
txGrid = ones(numDataSC, symPerSlot, 1);

% 标准 OFDM 调制：频域 → 时域
txWaveform = nrOFDMModulate(carrier, txGrid);

fprintf('  txGrid 尺寸    : %d × %d × %d\n', size(txGrid, 1), size(txGrid, 2), size(txGrid, 3));
fprintf('  txWaveform 长度: %d 个采样点\n\n', length(txWaveform));

% 重置信道，通过同一 TDL-D 信道
reset(channel);
rxWaveform = channel(txWaveform);

% 取第一根接收天线
rxWaveform = rxWaveform(:, 1);

% 标准 OFDM 解调：时域 → 频域
rxGrid = nrOFDMDemodulate(carrier, rxWaveform);

% 提取有效频带信道响应：H_active = rxGrid / txGrid（逐元素除法）
% txGrid 全为 1，因此 H_active = rxGrid
H_active = rxGrid(:, :, 1) ./ txGrid(:, :, 1);

% 计算 300 个激活子载波的平均功率（对所有子载波和符号取均值）
power_active = mean(abs(H_active).^2, 'all');

fprintf('  rxGrid 尺寸    : %d × %d\n', size(rxGrid, 1), size(rxGrid, 2));
fprintf('  H_active 尺寸  : %d × %d\n', size(H_active, 1), size(H_active, 2));
fprintf('\n');
fprintf('  >>> Test 2: 中心激活频带 (300 点) 平均能量 = %.6f\n', power_active);
fprintf('      预期值 ≈ 1.45（中心通带能量被异常放大）\n');
fprintf('      相对于全带宽的放大倍数 = %.4f×\n\n', power_active / power_full);

%% ── 综合裁决 ─────────────────────────────────────────────────────────────
fprintf('=================================================================\n');
fprintf('  综合裁决\n');
fprintf('=================================================================\n');
fprintf('  全频带 (512 点) 平均能量  : %.6f  （预期 ≈ 1.0）\n', power_full);
fprintf('  激活频带 (300 点) 平均能量 : %.6f  （预期 ≈ 1.45）\n', power_active);
fprintf('  能量放大比 (300/512 带宽)  : %.4f×\n', power_active / power_full);
fprintf('\n');

ratio_theory = Nfft / numDataSC;   % 512/300 ≈ 1.707（纯带宽比上限）
fprintf('  纯带宽比上限 (Nfft/N_active) = %d/%d = %.4f×\n', ...
        Nfft, numDataSC, ratio_theory);
fprintf('\n');

if power_full > 0.95 && power_full < 1.05
    fprintf('  [PASS] Test 1: 全频带能量守恒验证通过（偏差 < 5%%）\n');
else
    fprintf('  [FAIL] Test 1: 全频带能量守恒验证失败（偏差 >= 5%%）\n');
end

if power_active > 1.1
    fprintf('  [PASS] Test 2: 激活频带能量放大现象已确认（> 1.1×）\n');
    fprintf('         物理结论：nrTDLChannel 全局归一化导致局部频带虚高！\n');
else
    fprintf('  [INFO] Test 2: 激活频带能量未见明显放大（<= 1.1×）\n');
    fprintf('         可能原因：TDL-D 信道本身能量集中在直射径，边缘衰减不显著。\n');
end

fprintf('\n=================================================================\n');
fprintf('  验证完毕\n');
fprintf('=================================================================\n');
