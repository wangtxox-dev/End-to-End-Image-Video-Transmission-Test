function [ber, rx_bits] = sim_ofdm_worker(tx_bits_py, R, G, snr_dB)
% sim_ofdm_worker  供 Python matlabengine 调用的 OFDM + TDL-C 物理层函数
%
% 信道模型：5G NR OFDM + TDL-C 多径衰落信道
% 信道估计：完美信道估计（直接从无噪声接收网格计算 H = rxGrid_clean / txGrid）
% 均衡方式：逐子载波 MMSE 均衡
% LLR 加权：CSI scaling
%
% 输入:
%   tx_bits_py  - Python 传入的一维 0/1 数组 (K bits)
%   R           - 信道编码码率 (e.g. 1/8)
%   G           - 空口总比特数 (e.g. 786432)
%   snr_dB      - 信噪比 (dB)
%
% 输出:
%   ber         - 误比特率 (0 表示无误码)
%   rx_bits     - 解码后的 K 个比特 (int32 列向量)

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

    % 构建资源网格
    txGrid = zeros(numDataSC, numOFDMSym, 1);
    txSym_padded = [txSym; zeros(numDataSC * numOFDMSym - numSym, 1)];
    txGrid(:, :, 1) = reshape(txSym_padded, numDataSC, numOFDMSym);

    % OFDM 调制
    txWaveform = nrOFDMModulate(carrier, txGrid);

    % ── OFDM 系统信息 ─────────────────────────────────────────
    ofdmInfo   = nrOFDMInfo(carrier);
    sampleRate = ofdmInfo.SampleRate;

    % ── TDL-C 信道（多普勒归零，建立完美块衰落）─────────────
    % MaximumDopplerShift=0：信道在整个传输时长内绝对恒定，
    % 消除 ICI 和信道老化，建立理想块衰落（Block Fading）上限。
    channel = nrTDLChannel;
    channel.DelayProfile        = 'TDL-C';
    channel.DelaySpread         = 300e-9;
    channel.MaximumDopplerShift = 0;       % 绝对静止，消除多普勒
    channel.SampleRate          = sampleRate;
    channel.NumTransmitAntennas = 1;
    channel.NumReceiveAntennas  = 1;
    reset(channel);

    % 只通过信道一次，得到衰落后的时域波形
    rxWaveform_faded = channel(txWaveform);

    % ── 步骤1：时域加噪（允许用 measured）────────────────────
    rxWaveform_noisy = awgn(rxWaveform_faded, snr_dB, 'measured');

    % ── 步骤2：解调出纯净网格和含噪网格 ──────────────────────
    rxGrid_clean = nrOFDMDemodulate(carrier, rxWaveform_faded);
    rxGrid_noisy = nrOFDMDemodulate(carrier, rxWaveform_noisy);

    nCols_clean  = size(rxGrid_clean, 2);
    nCols_use    = min(nCols_clean, numOFDMSym);

    txGrid_2d       = txGrid(:, 1:nCols_use, 1);
    rxGrid_clean_2d = rxGrid_clean(:, 1:nCols_use, 1);

    % ── 步骤3：提取有效数据的 Mask ────────────────────────────
    nonzero_mask = abs(txGrid_2d) > 1e-10;

    % ── 步骤4：计算纯净频域数据子载波的真实信号方差 ──────────
    % 使用 var() 而非 mean(|.|²)，排除均值偏移（QPSK 均值=0，两者等价）
    sigPower_freq = mean(abs(rxGrid_clean_2d(nonzero_mask)).^2);

    % ── 步骤5：倒推最精准的频域 noiseVar ─────────────────────
    snr_linear = 10^(snr_dB / 10);
    noiseVar   = sigPower_freq / snr_linear;

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

    % ── OFDM 解调（含噪）─────────────────────────────────────
    nCols_actual = size(rxGrid_noisy, 2);
    nCols_eq     = min(nCols_actual, nCols_use);

    rxGrid_2d = rxGrid_noisy(:, 1:nCols_eq, 1);
    H_eq      = H_perfect(:, 1:nCols_eq);

    % MMSE 权重：W = H* / (|H|² + noiseVar)
    H_conj  = conj(H_eq);
    H_power = abs(H_eq).^2;
    W_mmse  = H_conj ./ (H_power + noiseVar);

    rxSym_eq = W_mmse .* rxGrid_2d;

    % CSI = 后验 SINR = |H|² / noiseVar（用于 LLR 加权）
    csi_grid = H_power / noiseVar;

    % ── 提取有效数据符号 ──────────────────────────────────────
    rxSym_vec = reshape(rxSym_eq, [], 1);
    csi_vec   = reshape(csi_grid, [], 1);

    numSym_avail = min(numSym, length(rxSym_vec));
    rxSym_vec    = rxSym_vec(1:numSym_avail);
    csi_vec      = csi_vec(1:numSym_avail);

    if numSym_avail < numSym
        rxSym_vec = [rxSym_vec; zeros(numSym - numSym_avail, 1)];
        csi_vec   = [csi_vec;   ones(numSym - numSym_avail, 1)];
    end

    % ── QPSK 软解调 ────────────────────────────────────────────
    % 使用 noiseVar=1，后续由 CSI scaling 控制软信息可靠性
    rxLLR_sym = nrSymbolDemodulate(rxSym_vec, modType, 1);

    % ── CSI 加权 ───────────────────────────────────────────────
    % QPSK 每符号 2 bit，csi 扩展 2 倍
    csi_expanded = repelem(csi_vec, 2);
    rxLLR_scaled = rxLLR_sym .* csi_expanded;

    % ── 速率恢复 + LDPC 解码 ───────────────────────────────────
    raterec = nrRateRecoverLDPC(rxLLR_scaled, K, R, rv, modType, nlayers, C);

    decBits_mat = nrLDPCDecode(raterec, bgn, 50);

    [rxBits_crc, ~] = nrCodeBlockDesegmentLDPC(decBits_mat, bgn, blkLen_with_crc);

    [rxBits, ~] = nrCRCDecode(rxBits_crc, crcType);

    % ── 性能统计 ───────────────────────────────────────────────
    [~, ber] = biterr(txBits, rxBits);
    rx_bits  = int32(rxBits(:));
end
