function [ber, rx_bits] = sim_awgn_worker(tx_bits_py, R, G, snr_dB)
% sim_awgn_worker  供 Python matlabengine 调用的纯物理层函数
%
% 输入:
%   tx_bits_py  - Python 传入的一维 0/1 数组 (K bits)
%   R           - 信道编码码率 (e.g. 1/8)
%   G           - 空口总比特数 (e.g. 256*256*3)
%   snr_dB      - 信噪比 (dB)
%
% 输出:
%   ber         - 误比特率 (0 表示无误码)
%   rx_bits     - 解码后的 K 个比特 (int32 列向量)

    % ── 打乱 RNG 状态，确保每次 SNR 点的 AWGN 噪声序列不同 ──────
    % 使用 'shuffle' 以系统时钟为种子，消除固化随机状态导致的确定性偏差
    rng('shuffle');

    % ── 数据转换 ──────────────────────────────────────────────
    txBits = double(tx_bits_py(:));   % 强制转为 MATLAB double 列向量
    K      = length(txBits);

    % ── 选择基图 (Base Graph) ──────────────────────────────────
    if (K <= 292) || (K <= 3824 && R <= 0.67) || R <= 0.25
        bgn = 2;
    else
        bgn = 1;
    end

    modType = 'QPSK';
    nlayers = 1;
    rv      = 0;
    noiseVar = 10^(-snr_dB / 10);

    % ── 发送端 (Tx) ────────────────────────────────────────────
    if K > 3824
        crcType = '24A';
    else
        crcType = '16';
    end

    txBits_crc      = nrCRCEncode(txBits, crcType);
    blkLen_with_crc = length(txBits_crc);

    cbsMat  = nrCodeBlockSegmentLDPC(txBits_crc, bgn);
    [~, C]  = size(cbsMat);

    encMat  = nrLDPCEncode(cbsMat, bgn);

    % 速率匹配：传总长 G，内部自动分配各码块 E 值
    txRM_vec = nrRateMatchLDPC(encMat, G, rv, modType, nlayers);

    txSig = nrSymbolModulate(txRM_vec, modType);

    % ── 信道 (AWGN) ────────────────────────────────────────────
    rxSig = awgn(txSig, snr_dB, 'measured');

    % ── 接收端 (Rx) ────────────────────────────────────────────
    rxLLR_vec = nrSymbolDemodulate(rxSig, modType, noiseVar);

    raterec   = nrRateRecoverLDPC(rxLLR_vec, K, R, rv, modType, nlayers, C);

    decBits_mat = nrLDPCDecode(raterec, bgn, 50);

    [rxBits_crc, ~] = nrCodeBlockDesegmentLDPC(decBits_mat, bgn, blkLen_with_crc);

    [rxBits, ~] = nrCRCDecode(rxBits_crc, crcType);

    % ── 性能统计 ───────────────────────────────────────────────
    [~, ber] = biterr(txBits, rxBits);
    rx_bits  = int32(rxBits(:));
end
