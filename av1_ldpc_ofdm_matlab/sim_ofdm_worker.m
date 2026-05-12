function [ber, rx_bits, H_energy] = sim_ofdm_worker(tx_bits_py, R, G, snr_dB, channelModel, delaySpread)
% sim_ofdm_worker  供 Python matlabengine 调用的 OFDM 物理层函数
%
% 信道模型：由 channelModel 参数决定
%   'AWGN'  — 纯加性高斯白噪声，跳过多径衰落
%   'TDL-*' — 5G NR TDL 多径衰落信道（如 TDL-C、TDL-D）
% 信道估计：完美信道估计（直接从无噪声接收网格计算 H = rxGrid_clean / txGrid）
% 均衡方式：逐子载波 Unbiased MMSE 均衡
% LLR 计算：等效噪声方差直接喂给 nrSymbolDemodulate，无需手动 CSI 加权
%
% 输入:
%   tx_bits_py   - Python 传入的一维 0/1 数组 (K bits)
%   R            - 信道编码码率 (e.g. 1/8)
%   G            - 空口总比特数 (e.g. 786432)
%   snr_dB       - 信噪比 (dB)
%   channelModel - 信道模型字符串，如 'AWGN'、'TDL-C'、'TDL-D'
%   delaySpread  - 时延扩展 (秒)，AWGN 时忽略，如 100e-9
%
% 输出:
%   ber         - 误比特率 (0 表示无误码)
%   rx_bits     - 解码后的 K 个比特 (int32 列向量)
%   H_energy    - 本次随机快照的真实平均信道能量 (AWGN 恒为 1.0)

    rng('shuffle');

    % ── 数据转换 ──────────────────────────────────────────────
    txBits = double(tx_bits_py(:));
    K      = length(txBits);

    % ── 选择基图 ──────────────────────────────────────────────
    if (K <= 292) || (K <= 3824 && R <= 0.67) || R <= 0.25
        bgn = 2;
    else
        bgn = 1;
    end

    modType = 'QPSK';
    nlayers = 1;
    rv      = 0;

    % ── 发送端 ────────────────────────────────────────────────
    if K > 3824
        crcType = '24A';
    else
        crcType = '16';
    end

    txBits_crc      = nrCRCEncode(txBits, crcType);
    blkLen_with_crc = length(txBits_crc);
    cbsMat          = nrCodeBlockSegmentLDPC(txBits_crc, bgn);
    [~, C]          = size(cbsMat);
    encMat          = nrLDPCEncode(cbsMat, bgn);
    txRM_vec        = nrRateMatchLDPC(encMat, G, rv, modType, nlayers);
    txSym           = nrSymbolModulate(txRM_vec, modType);
    numSym          = length(txSym);   % = G/2

    % ── 发射端符号级交织 ──────────────────────────────────────
    % 固定种子 rng(42)，生成随机置换索引，打散符号顺序
    % 目的：将连片深度衰落分散到不同 LDPC 码字位置，增强纠错能力
    % 【RNG 隔离】保存当前全局 RNG 状态，用完后立即恢复，
    % 防止 rng(42) 污染后续信道生成所需的随机流。
    rng_state_saved = rng();   % 保存全局 RNG 快照
    rng(42);
    intrlv_idx = randperm(numSym);
    rng(rng_state_saved);      % 立即恢复，全局随机流不受影响
    txSym_intrlv = txSym(intrlv_idx);   % 打乱后的符号序列

    % ── OFDM 配置 ─────────────────────────────────────────────
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = 30;   % 30 kHz SCS (mu=1)
    carrier.NSizeGrid         = 25;   % 25 RB = 300 数据子载波/符号
    carrier.NStartGrid        = 0;

    numDataSC  = carrier.NSizeGrid * 12;   % 300
    symPerSlot = 14;

    % 计算需要多少个完整 slot
    numSlots   = ceil(numSym / (numDataSC * symPerSlot));
    numOFDMSym = numSlots * symPerSlot;

    % 构建资源网格（使用交织后的符号）
    txGrid = zeros(numDataSC, numOFDMSym, 1);
    txSym_padded = [txSym_intrlv; zeros(numDataSC * numOFDMSym - numSym, 1)];
    txGrid(:, :, 1) = reshape(txSym_padded, numDataSC, numOFDMSym);

    % OFDM 调制
    txWaveform = nrOFDMModulate(carrier, txGrid);

    % ── OFDM 系统信息 ─────────────────────────────────────────
    ofdmInfo   = nrOFDMInfo(carrier);
    sampleRate = ofdmInfo.SampleRate;

    % ── 信道传播 ──────────────────────────────────────────────
    if strcmp(channelModel, 'AWGN')
        % AWGN 直通：跳过多径衰落，时域波形不变
        rxWaveform_faded = txWaveform;
    else
        % TDL-* 多径衰落信道
        % MaximumDopplerShift=0：信道在整个传输时长内绝对恒定，
        % 消除 ICI 和信道老化，建立理想块衰落（Block Fading）上限。
        channel = nrTDLChannel;
        channel.DelayProfile        = channelModel;   % 如 'TDL-C'、'TDL-D'
        channel.DelaySpread         = delaySpread;    % 如 100e-9
        channel.MaximumDopplerShift = 0;              % 绝对静止，消除多普勒
        channel.SampleRate          = sampleRate;
        channel.NumTransmitAntennas = 1;
        channel.NumReceiveAntennas  = 1;
        % 【关键修复】使用全局随机流而非对象内部固定状态。
        % reset(channel) 会将信道对象的内部 RNG 重置到确定性初始值，
        % 导致每次调用产生完全相同的衰落系数（H_energy 永远是 1.4714）。
        % 改为 'Global stream' 后，信道从 MATLAB 全局随机流取随机数，
        % 而全局流已在函数入口由 rng('shuffle') 随机化，
        % 从而保证每次快照得到真正独立的随机衰落系数。
        channel.RandomStream = 'Global stream';
        % 不调用 reset(channel)，避免锁死内部随机状态

        % 只通过信道一次，得到衰落后的时域波形
        rxWaveform_faded = channel(txWaveform);
    end

    % ── 步骤1：建立绝对底噪基准（发端对齐）──────────────────
    % 发射端 QPSK 符号能量期望 = 1.0（nrSymbolModulate 归一化输出）
    % 频域噪声方差直接由输入 snr_dB 决定，与接收信号功率无关
    snr_linear = 10^(snr_dB / 10);
    noiseVar   = 1.0 / snr_linear;   % 频域每子载波噪声方差

    % ── 步骤2：时域精确加噪（补偿 nrOFDMModulate 的 FFT 缩放）─
    % nrOFDMModulate 内部执行 IFFT 并除以 sqrt(Nfft)，
    % 使时域信号能量 = 频域能量 / Nfft。
    % 根据 Parseval 定理，时域噪声方差需同步缩小 Nfft 倍，
    % 才能保证解调后频域噪声方差 = noiseVar。
    Nfft     = double(ofdmInfo.Nfft);
    N0_time  = noiseVar / Nfft;      % 时域复高斯噪声方差（每维 N0_time/2）

    noise_time = (randn(size(rxWaveform_faded)) + ...
                  1j * randn(size(rxWaveform_faded))) * sqrt(N0_time / 2);
    rxWaveform_noisy = rxWaveform_faded + noise_time;

    % ── 步骤3：解调出纯净网格和含噪网格 ──────────────────────
    rxGrid_noisy = nrOFDMDemodulate(carrier, rxWaveform_noisy);

    % ── 完美信道估计 ──────────────────────────────────────────
    if strcmp(channelModel, 'AWGN')
        % AWGN：无多径失真，H 恒为全1矩阵
        nCols_noisy  = size(rxGrid_noisy, 2);
        nCols_use    = min(nCols_noisy, numOFDMSym);
        H_perfect    = ones(numDataSC, nCols_use);
    else
        % TDL-*：从无噪声接收网格提取完美 H
        rxGrid_clean = nrOFDMDemodulate(carrier, rxWaveform_faded);

        nCols_clean  = size(rxGrid_clean, 2);
        nCols_use    = min(nCols_clean, numOFDMSym);

        txGrid_2d       = txGrid(:, 1:nCols_use, 1);
        rxGrid_clean_2d = rxGrid_clean(:, 1:nCols_use, 1);

        % ── 步骤4：提取有效数据的 Mask ────────────────────────────
        nonzero_mask = abs(txGrid_2d) > 1e-10;
        % noiseVar 已在步骤1确定，无需从接收信号反推，直接用于后续均衡

        % ── 完美信道估计：H = rxGrid_clean / txGrid ───────────────
        H_perfect    = zeros(numDataSC, nCols_use);
        H_perfect(nonzero_mask) = rxGrid_clean_2d(nonzero_mask) ./ txGrid_2d(nonzero_mask);

        % 对补零列用前一列 H 填充
        for col = 1:nCols_use
            zero_sc = ~nonzero_mask(:, col);
            if any(zero_sc) && col > 1
                H_perfect(zero_sc, col) = H_perfect(zero_sc, col-1);
            elseif any(zero_sc) && col == 1
                H_perfect(zero_sc, col) = 1.0;
            end
        end
    end

    % ── OFDM 解调（含噪）─────────────────────────────────────
    nCols_actual = size(rxGrid_noisy, 2);
    nCols_eq     = min(nCols_actual, nCols_use);

    rxGrid_2d = rxGrid_noisy(:, 1:nCols_eq, 1);
    H_eq      = H_perfect(:, 1:nCols_eq);

    % ── Unbiased MMSE 均衡 ────────────────────────────────────
    % 标准 MMSE 权重：W = H* / (|H|² + N0)
    % 均衡后信号：y_eq = g * s + noise，其中等效增益 g = |H|² / (|H|² + N0)
    % Unbiased 步骤：将 y_eq 除以 g，强行把振幅拉回 1.0
    % 等效 SINR：gamma = g / (1 - g) = |H|² / N0
    % 等效噪声方差：sigma²_eff = 1 / gamma = N0 / |H|²
    H_conj  = conj(H_eq);
    H_power = abs(H_eq).^2;

    % 等效增益 g = |H|² / (|H|² + N0)，范围 (0, 1)
    g_eff = H_power ./ (H_power + noiseVar);

    % MMSE 均衡（有偏）
    W_mmse   = H_conj ./ (H_power + noiseVar);
    rxSym_biased = W_mmse .* rxGrid_2d;

    % 消除偏置：除以 g_eff，振幅恢复为 1.0
    % 对 g_eff 极小的子载波（深度衰落）做保护，避免除以零
    g_safe   = max(g_eff, 1e-10);
    rxSym_eq = rxSym_biased ./ g_safe;

    % 等效噪声方差：sigma²_eff = (1 - g) / g = N0 / |H|²
    % 此方差已天然包含信道可靠度信息，无需额外 CSI 加权
    noiseVar_eff = (1 - g_eff) ./ g_safe;   % 与 N0 / |H|² 等价

    % ── 提取有效数据符号 ──────────────────────────────────────
    rxSym_vec      = reshape(rxSym_eq,      [], 1);
    noiseVar_eff_v = reshape(noiseVar_eff,  [], 1);

    numSym_avail = min(numSym, length(rxSym_vec));
    rxSym_vec      = rxSym_vec(1:numSym_avail);
    noiseVar_eff_v = noiseVar_eff_v(1:numSym_avail);

    if numSym_avail < numSym
        % 补零兜底符号：等效噪声方差设为极大值，使 LLR 强制趋近于 0
        rxSym_vec      = [rxSym_vec;      zeros(numSym - numSym_avail, 1)];
        noiseVar_eff_v = [noiseVar_eff_v; 1e8 * ones(numSym - numSym_avail, 1)];
    end

    % ── 接收端符号级解交织 ────────────────────────────────────
    % 使用完全相同的种子 rng(42) 重建置换索引，计算反向索引还原原始顺序
    % 【RNG 隔离】同样保存/恢复全局状态，确保解交织不污染后续随机流。
    rng_state_saved2 = rng();   % 保存全局 RNG 快照
    rng(42);
    intrlv_idx_rx = randperm(numSym);
    rng(rng_state_saved2);      % 立即恢复
    deintrlv_idx  = zeros(1, numSym);
    deintrlv_idx(intrlv_idx_rx) = 1:numSym;   % 反向索引

    rxSym_vec      = rxSym_vec(deintrlv_idx);        % 还原符号顺序
    noiseVar_eff_v = noiseVar_eff_v(deintrlv_idx);   % 同步还原噪声方差顺序

    % ── QPSK 软解调（Unbiased MMSE）──────────────────────────
    % nrSymbolDemodulate 仅接受标量噪声方差，无法直接传入向量。
    % 对于 QPSK（Gray 映射，归一化星座点 ±1/√2 + j·±1/√2），
    % 最优 LLR 解析式为：
    %   LLR_bit0 = 2*√2 * real(y) / sigma²_eff
    %   LLR_bit1 = 2*√2 * imag(y) / sigma²_eff
    % nrSymbolModulate 输出的 QPSK 星座点幅值为 1/√2，
    % 因此判决边界处斜率因子 = 2 * (1/√2) / (1/2) = 2√2。
    % 等价地：LLR = 2 * real/imag(y) / (sigma²_eff / √2 * √2)
    % 简化后直接用 nrSymbolDemodulate 标量=1 再乘以 1/sigma²_eff 向量：
    %   nrSymbolDemodulate(y, 'QPSK', 1) 输出 = 2√2 * real/imag(y)
    %   再除以 sigma²_eff 即得正确 LLR。
    llr_unit     = nrSymbolDemodulate(rxSym_vec, modType, 1);   % 假设 sigma²=1
    % llr_unit 长度 = 2*numSym（每符号 2 bit），需将 noiseVar_eff_v 扩展 2 倍
    noiseVar_eff_bits = repelem(noiseVar_eff_v, 2);
    rxLLR_scaled = llr_unit ./ noiseVar_eff_bits;

    % ── 速率恢复 + LDPC 解码 ───────────────────────────────────
    raterec = nrRateRecoverLDPC(rxLLR_scaled, K, R, rv, modType, nlayers, C);

    decBits_mat = nrLDPCDecode(raterec, bgn, 50);

    [rxBits_crc, ~] = nrCodeBlockDesegmentLDPC(decBits_mat, bgn, blkLen_with_crc);

    [rxBits, ~] = nrCRCDecode(rxBits_crc, crcType);

    % ── 性能统计 ───────────────────────────────────────────────
    [~, ber] = biterr(txBits, rxBits);
    rx_bits  = int32(rxBits(:));

    % 提取本次随机快照的真实平均能量
    if strcmp(channelModel, 'AWGN')
        H_energy = 1.0;
    else
        H_energy = mean(abs(H_perfect).^2, 'all');
    end
end
