function ber_array = sim_5g_awgn_link(R, G, snr_dB_array)
    % 5G NR LDPC 链路级仿真引擎（正确版）
    % 关键接口规则：
    %   Tx: nrRateMatchLDPC(encMat, G, rv, mod, nLayers)  -- 传总长G，批量处理
    %   Rx: nrRateRecoverLDPC(rxLLR, K, R, rv, mod, nLayers, C) -- 传原始K（不含CRC）
    
    K = round(G * R); 
    if (K <= 292) || (K <= 3824 && R <= 0.67) || R <= 0.25
        bgn = 2; 
    else
        bgn = 1; 
    end
    
    modType = 'QPSK';
    nlayers = 1;
    rv = 0; 
        
    ber_array = zeros(size(snr_dB_array));
    fprintf('开始仿真 R = %-5s (信源 K = %-6d, 空口 G = %d, 使用 BG%d)\n', strtrim(rats(R)), K, G, bgn);
    
    for i = 1:length(snr_dB_array)
        snr = snr_dB_array(i);
        noiseVar = 10^(-snr/10); 
        
        % ================= 发送端 (Tx) =================
        % txBits = randi([0 1], K, 1);

        % 生成 1 的概率为 80%, 0 的概率为 20% 的信源
        p_one = 0.5; 
        txBits = double(rand(K, 1) < p_one);
        
        if K > 3824
            crcType = '24A';
        else
            crcType = '16';
        end
        txBits_crc = nrCRCEncode(txBits, crcType);
        blkLen_with_crc = length(txBits_crc); 
        
        % 码块分割：返回 K_cb x C 矩阵
        cbsMat = nrCodeBlockSegmentLDPC(txBits_crc, bgn);
        [~, C] = size(cbsMat);
        
        % 批量编码：K_cb x C → N_ldpc x C
        encMat = nrLDPCEncode(cbsMat, bgn);
        
        % 批量速率匹配：传总长G，内部自动分配各码块E值
        txRM_vec = nrRateMatchLDPC(encMat, G, rv, modType, nlayers);
        
        txSig = nrSymbolModulate(txRM_vec, modType);
        rxSig = awgn(txSig, snr, 'measured');
        
        % ================= 接收端 (Rx) =================
        rxLLR_vec = nrSymbolDemodulate(rxSig, modType, noiseVar);
        
        % 速率恢复：传原始K（不含CRC），输出 N_ldpc x C
        raterec = nrRateRecoverLDPC(rxLLR_vec, K, R, rv, modType, nlayers, C);
        
        % 批量解码：N_ldpc x C → K_cb x C
        decBits_mat = nrLDPCDecode(raterec, bgn, 50);
        
        % 码块解分割：K_cb x C → 含CRC的比特流
        [rxBits_crc, ~] = nrCodeBlockDesegmentLDPC(decBits_mat, bgn, blkLen_with_crc);
        
        % 剥离 CRC，恢复原始 K 个信源比特
        [rxBits, ~] = nrCRCDecode(rxBits_crc, crcType);
        
        % ================= 性能统计 =================
        [~, ber_array(i)] = biterr(txBits, rxBits);
        
        % 取消 if 判断，无条件打印每个 SNR 点的 BER
        fprintf('  SNR = %6.2f dB | BER = %.2e\n', snr, ber_array(i));
    end
end
