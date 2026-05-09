% debug_ofdm_link.m  —  最小化诊断：只测试 H_est 尺度
clear; clc;
rng(42);

fprintf('=== 最小化诊断 ===\n');

% 最小参数
carrier=nrCarrierConfig;
carrier.SubcarrierSpacing=30;
carrier.NSizeGrid=25;

numDataSC=300;
numOFDMSym=14;  % 1 slot

% 发送已知符号（全1）
txGrid=ones(numDataSC,numOFDMSym,1);
txWaveform=nrOFDMModulate(carrier,txGrid);

ofdmInfo=nrOFDMInfo(carrier);
sampleRate=ofdmInfo.SampleRate;
Nfft=ofdmInfo.Nfft;
fprintf('Nfft=%d, sampleRate=%.2e\n',Nfft,sampleRate);
fprintf('txWaveform 功率: %.6f\n',mean(abs(txWaveform).^2));

% TDL-C
channel=nrTDLChannel;
channel.DelayProfile='TDL-C';
channel.DelaySpread=300e-9;
channel.MaximumDopplerShift=30;
channel.SampleRate=sampleRate;
channel.NumTransmitAntennas=1;
channel.NumReceiveAntennas=1;
reset(channel);

[rxWave_faded,pathGains,sampleTimes]=channel(txWaveform);
rxGrid_clean=nrOFDMDemodulate(carrier,rxWave_faded);

pathFilters=getPathFilters(channel);
H_est=nrPerfectChannelEstimate(carrier,pathGains,pathFilters,0,sampleTimes);

fprintf('\ntxGrid 功率: %.4f\n',mean(abs(txGrid(:)).^2));
fprintf('rxGrid_clean 功率: %.4f\n',mean(abs(rxGrid_clean(:)).^2));
fprintf('H_est 均值幅度: %.4f\n',mean(abs(H_est(:))));
fprintf('H_est 尺寸: %dx%d\n',size(H_est,1),size(H_est,2));
fprintf('rxGrid_clean 尺寸: %dx%d\n',size(rxGrid_clean,1),size(rxGrid_clean,2));

% 关键：rxGrid_clean = H_est * txGrid（逐元素）吗？
nC=min(size(rxGrid_clean,2),size(H_est,2));
H_e=H_est(:,1:nC,1,1);
rxG=rxGrid_clean(:,1:nC,1);
txG=txGrid(:,1:nC,1);

% 因为 txGrid=1，所以 rxGrid_clean 应该 ≈ H_est（如果尺度一致）
fprintf('\nrxGrid_clean 前3个元素: ');
fprintf('%.4f%+.4fi  ',rxG(1:3));
fprintf('\nH_est 前3个元素:        ');
fprintf('%.4f%+.4fi  ',H_e(1:3));
fprintf('\n比值 rxGrid/H_est 前3个: ');
fprintf('%.4f  ',abs(rxG(1:3)./H_e(1:3)));
fprintf('\n');

% 用 H_est 做 MMSE 均衡（SNR=20dB）
snr_dB=20;
snr_lin=10^(snr_dB/10);
noiseVar=1/snr_lin;

rxWave_noisy=awgn(rxWave_faded,snr_dB,'measured');
rxGrid_noisy=nrOFDMDemodulate(carrier,rxWave_noisy);

[rxSym_eq,csi]=nrEqualizeMMSE(rxGrid_noisy(:,1:nC,:),H_est(:,1:nC,:,:),noiseVar);
rxSym_vec=reshape(rxSym_eq(:,:,1),[],1);
fprintf('\n均衡后符号功率: %.4f (期望: ~1)\n',mean(abs(rxSym_vec).^2));
fprintf('均衡后前3个符号: ');
fprintf('%.4f%+.4fi  ',rxSym_vec(1:3));
fprintf('\n发送前3个符号:   ');
fprintf('%.4f%+.4fi  ',txGrid(1:3));
fprintf('\n');

fprintf('\n=== 诊断完成 ===\n');
