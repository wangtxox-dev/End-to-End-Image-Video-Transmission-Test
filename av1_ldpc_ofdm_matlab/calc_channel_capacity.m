function C = calc_channel_capacity(snr_dB, channelModel, delaySpread)
% calc_channel_capacity  通用信道遍历容量计算函数
%
% 支持信道模型：
%   'AWGN'      — 直接套用香农公式，极速返回
%   'TDL-*'     — 通过 1000 次蒙特卡洛仿真计算遍历容量
%
% 输入:
%   snr_dB       - 信噪比 (dB)，标量
%   channelModel - 信道模型字符串，如 'AWGN'、'TDL-C'、'TDL-D' 等
%   delaySpread  - 时延扩展 (秒)，AWGN 时忽略，如 100e-9
%
% 输出:
%   C            - 遍历容量 (bits/s/Hz)，标量

    snr_linear = 10^(snr_dB / 10);

    % ── AWGN 快速路径 ────────────────────────────────────────────
    if strcmp(channelModel, 'AWGN')
        C = mean(log2(1 + 10.^(snr_dB / 10)));
        return;
    end

    % ── TDL-* 蒙特卡洛路径 ───────────────────────────────────────
    numMC = 1000;   % 蒙特卡洛次数

    % ── OFDM 载波配置（与 sim_ofdm_worker.m 保持一致）──────────
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = 30;   % 30 kHz SCS (mu=1)
    carrier.NSizeGrid         = 25;   % 25 RB = 300 数据子载波/符号
    carrier.NStartGrid        = 0;

    numDataSC  = carrier.NSizeGrid * 12;   % 300
    symPerSlot = 14;
    numOFDMSym = symPerSlot;               % 单 slot 足够采样信道

    % ── 构造一个全1参考网格（用于提取 H）──────────────────────
    txGrid_ref = ones(numDataSC, numOFDMSym, 1);

    ofdmInfo   = nrOFDMInfo(carrier);
    sampleRate = ofdmInfo.SampleRate;

    % ── TDL 信道配置（按传入参数动态设置）──────────────────────
    channel = nrTDLChannel;
    channel.DelayProfile        = channelModel;   % 如 'TDL-C'、'TDL-D'
    channel.DelaySpread         = delaySpread;    % 如 100e-9
    channel.MaximumDopplerShift = 0;              % 块衰落，与 worker 一致
    channel.SampleRate          = sampleRate;
    channel.NumTransmitAntennas = 1;
    channel.NumReceiveAntennas  = 1;
    channel.RandomStream = 'mt19937ar with seed'; % 【新增】配置独立随机数流

    % ── 蒙特卡洛主循环 ──────────────────────────────────────────
    cap_accum = 0.0;

    for mc = 1:numMC
        % 强制释放对象并更改种子，确保每次获得绝对独立的衰落实现
        release(channel);
        channel.Seed = mc;

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
        H = rxGrid_2d ./ txGrid_2d;

        % 遍历容量：C = mean(log2(1 + |H|^2 * SNR))
        cap_mc = mean(mean(log2(1 + abs(H).^2 * snr_linear)));
        cap_accum = cap_accum + cap_mc;
    end

    C = cap_accum / numMC;
end
