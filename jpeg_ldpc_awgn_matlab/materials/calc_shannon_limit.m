function snr_limit_dB = calc_shannon_limit(R_array, Q)
    % 计算给定码率和调制阶数下的香农极限 SNR (dB)
    snr_limit_linear = 2.^(R_array * Q) - 1;
    snr_limit_dB = 10 * log10(snr_limit_linear);
end
