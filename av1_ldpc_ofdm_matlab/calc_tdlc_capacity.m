function C = calc_tdlc_capacity(snr_dB)
% calc_tdlc_capacity  通过蒙特卡洛仿真计算 TDL-C 信道的平均遍历容量
%
% 信道参数：TDL-C, DelaySpread=100ns, MaximumDopplerShift=0（块衰落）
% 仿真次数：1000 次独立信道实现
% 容量公式：C = mean(log2(1 + |H|^2 * SNR_linear))
%
% 输入:
%   snr_dB  - 信噪比 (dB)，标量
%
% 输出:
%   C       - TDL-C 遍历容量 (bits/s/Hz)，标量

    numMC = 1000;   % 蒙特卡洛次数

    snr_linear = 10^(snr_dB / 10);

    % ── OFDM 载波配置（与 sim_ofdm_worker.m 保持一致）──────────
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = 30;   % 30 kHz SCS (mu=1)
    carrier.NSizeGrid         = 25;   % 25 RB = 300 数据子载波/符号
    carrier.NStartGrid        = 0;

    numDataSC  = carrier.NSizeGrid * 12;   % 300
    symPerSlot = 14;
    numOFDMSym = symPerSlot;               % 单 slot 足够采样信道

    % ── 构造一个全1参考网格（用于提取 H）──────────────────────
    % 发送全1 QPSK 符号（幅度1），方便直接用 rxGrid_clean/txGrid 得到 H
    txGrid_ref = ones(numDataSC, numOFDMSym, 1);

    ofdmInfo   = nrOFDMInfo(carrier);
    sampleRate = ofdmInfo.SampleRate;

    % ── TDL-C 信道配置 ──────────────────────────────────────────
    channel = nrTDLChannel;
    channel.DelayProfile        = 'TDL-C';
    channel.DelaySpread         = 100e-9;
    channel.MaximumDopplerShift = 0;       % 块衰落，与 worker 一致
    channel.SampleRate          = sampleRate;
    channel.NumTransmitAntennas = 1;
    channel.NumReceiveAntennas  = 1;

    % ── 蒙特卡洛主循环 ──────────────────────────────────────────
    cap_accum = 0.0;

    for mc = 1:numMC
        % 每次重置信道，获得独立衰落实现
        reset(channel);

        % OFDM 调制参考波形
        txWaveform = nrOFDMModulate(carrier, txGrid_ref);

        % 通过信道（无噪声，仅获取衰落响应）
        rxWaveform_faded = channel(txWaveform);

        % OFDM 解调，得到纯净频域接收网格
        rxGrid_clean = nrOFDMDemodulate(carrier, rxWaveform_faded);

        nCols = min(size(rxGrid_clean, 2), numOFDMSym);
        rxGrid_2d = rxGrid_clean(:, 1:nCols, 1);
        txGrid_2d = txGrid_ref(:, 1:nCols, 1);

        % 完美信道估计：H = rxGrid_clean / txGrid（txGrid 全1，直接取 rxGrid）
        H = rxGrid_2d ./ txGrid_2d;   % txGrid_ref 全1，等价于直接取 rxGrid_2d

        % 遍历容量：C = mean(log2(1 + |H|^2 * SNR))
        cap_mc = mean(mean(log2(1 + abs(H).^2 * snr_linear)));
        cap_accum = cap_accum + cap_mc;
    end

    C = cap_accum / numMC;
end
