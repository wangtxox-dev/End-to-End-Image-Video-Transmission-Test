% 主控脚本：5G NR 任意码率 LDPC 性能深度剖析 (严格锁定空口比特数)
clear; clc; close all;

%% 1. 实验参数设置
rates = [1/8, 1/16, 1/20, 1/24, 1/64];
Q = 2; % QPSK
G = 512 * 512 * 2; % 严格锁定的空口传输比特数: 196608

% 生成码率标签（分数字符串，适用于任意码率）
rate_labels = cell(length(rates), 1);
for idx = 1:length(rates)
    R = rates(idx);
    % 用 rats() 获取最简分数字符串
    lbl = strtrim(rats(R, 8));
    rate_labels{idx} = lbl;
end

snr_ranges  = cell(length(rates), 1);
ber_results = cell(length(rates), 1);
actual_limits = zeros(length(rates), 1);
rep_limits    = zeros(length(rates), 1);

%% 2. 计算真实香农极限
shannon_limits = calc_shannon_limit(rates, Q);

%% 3. 执行仿真
for idx = 1:length(rates)
    R = rates(idx);
    
    % 计算 5G NR 的"重复理论极限"
    % BG2 母码率约 1/5；低于 1/5 时底层做 LLR 软合并
    if R < 1/5
        base_limit  = calc_shannon_limit(1/5, Q);
        rep_factor  = (1/5) / R;
        rep_limits(idx) = base_limit - 10 * log10(rep_factor);
    else
        rep_limits(idx) = shannon_limits(idx);
    end
    
    % 智能设定 SNR 扫描范围
    % 起点：理论极限 - 0.5 dB；终点：理论极限 + 2.5 dB（保证 BER=0 后至少 3 个点）
    start_snr = floor(rep_limits(idx) * 10) / 10 - 0.5;
    end_snr   = ceil(rep_limits(idx)  * 10) / 10 + 2.5;
    snr_ranges{idx} = start_snr : 0.1 : end_snr;
    
    % 调用 5G 引擎
    ber_results{idx} = sim_5g_awgn_link(R, G, snr_ranges{idx});
    
    % 寻找实际稳定极限 (最后一个误码点 + 0.2 dB 裕度)
    last_err_idx = find(ber_results{idx} > 0, 1, 'last');
    if ~isempty(last_err_idx) && (last_err_idx + 2 <= length(snr_ranges{idx}))
        actual_limits(idx) = snr_ranges{idx}(last_err_idx + 2);
    else
        actual_limits(idx) = NaN;
    end
    
    % ---- BER 抖动检测 ----
    first_zero_idx = find(ber_results{idx} == 0, 1, 'first');
    if ~isempty(first_zero_idx)
        check_end = min(first_zero_idx + 3, length(ber_results{idx}));
        post_zero = ber_results{idx}(first_zero_idx + 1 : check_end);
        if any(post_zero > 0)
            fprintf('*** WARNING: BER fluctuation detected at Rate %s! (BER bounced after first zero at SNR=%.1f dB) ***\n', ...
                rate_labels{idx}, snr_ranges{idx}(first_zero_idx));
        end
    end
end

%% 4. 绘制图表
figure('Position', [100, 100, 900, 600]);
hold on; grid on;
colors = lines(length(rates));

for idx = 1:length(rates)
    % 将 BER=0 替换为 NaN 以避免 semilogy 报错
    ber_plot = ber_results{idx};
    ber_plot(ber_plot == 0) = NaN;
    semilogy(snr_ranges{idx}, ber_plot, ...
        'Color', colors(idx,:), 'LineWidth', 2, 'Marker', '.', ...
        'DisplayName', sprintf('R = %s', rate_labels{idx}));
    xline(shannon_limits(idx), '--', 'Color', colors(idx,:), 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

set(gca, 'YScale', 'log'); ylim([1e-6, 1]);
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('BER', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('5G NR LDPC 任意码率性能 vs 香农极限 (G = %d)', G), 'FontSize', 14);
legend('Location', 'southwest');
hold off;

%% 5. 打印终极算账表格
fprintf('\n============== 5G NR 任意码率性能深度剖析 ==============\n');
fprintf(' 码率    | 信源比特(K) | 真实香农极限 | 5G理论极限(含重复) | 实际仿真极限 | 总差距(Gap)\n');
fprintf('------------------------------------------------------------------------------------\n');
for idx = 1:length(rates)
    K_val     = round(G * rates(idx));
    gap_total = actual_limits(idx) - shannon_limits(idx);
    fprintf(' %-7s |   %-8d  |  %7.2f dB  |     %7.2f dB     |  %7.2f dB  |  %6.2f dB\n', ...
        rate_labels{idx}, K_val, shannon_limits(idx), rep_limits(idx), actual_limits(idx), gap_total);
end
fprintf('====================================================================================\n');
