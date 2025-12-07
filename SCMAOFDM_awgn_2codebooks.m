function SCMA_OFDM_AWGN_compare_codebooks()
% Compare two SCMA codebooks over a pure AWGN channel (no fading, no CFO).
% - Codebook 1: "yours"
% - Codebook 2: "his"
%
% SCMA-OFDM structure kept similar to previous code:
%   - 4 resources, 4 codewords, 6 users
%   - OFDM with Nfft = 1024, Ncp = 32, interleaved mapping
%   - Perfect CSI = 1 (flat channel), no CFO, no phase noise, no SCO, no timing error
%   - AWGN only
%
% External dependency: scmadec(y, CB, h, N0_eff, Niter)

clc; close all;
% rng(2025);   % Uncomment for repeatable results

%% -------- Define two SCMA codebooks (4x4x6 each) --------
CB1 = zeros(4,4,6);   % your codebook
CB2 = zeros(4,4,6);   % his codebook

% ----- Your codebook -----
CB1(:,:,1) = [ ...
     0.4860+1j*0.7796  -0.2844+1j*0.7570   0.2923-1j*0.7594  -0.4770-1j*0.7546; ...
     0                  0                  0                  0; ...
     0.4270+1j*0.2213  -0.5341+1j*0.0822   0.5329-1j*0.0823  -0.4161-1j*0.2359; ...
     0                  0                  0                  0 ];

CB1(:,:,2) = [ ...
     0                  0                  0                  0; ...
     0.5482+1j*0.0589  -0.4082-1j*0.0441   0.4221+1j*0.0443  -0.5536-1j*0.0587; ...
     0                  0                  0                  0; ...
    -0.2544+1j*0.2930  -0.7828+1j*0.8851   0.7798-1j*0.8536   0.2745-1j*0.2800 ];

CB1(:,:,3) = [ ...
     0.3165-1j*0.6030   0.3219-1j*0.2866  -0.3276+1j*0.3023  -0.3112+1j*0.5885; ...
    -0.2928-1j*0.6702   0.8058+1j*0.3427  -0.8203-1j*0.3557   0.3103+1j*0.6851; ...
     0                  0                  0                  0; ...
     0                  0                  0                  0 ];

CB1(:,:,4) = [ ...
     0                  0                  0                  0; ...
     0                  0                  0                  0; ...
    -0.5496+1j*0.7374  -0.5721+1j*0.6638   0.4314-1j*0.5480   0.4564-1j*0.4795; ...
    -0.6670-1j*0.6866   0.6563+1j*0.7012  -0.4234-1j*0.6033   0.4306+1j*0.5856 ];

CB1(:,:,5) = [ ...
    -0.8126-1j*0.3453   0.7364-1j*0.4534  -0.7468+1j*0.4411   0.8132+1j*0.3525; ...
     0                  0                  0                  0; ...
     0                  0                  0                  0; ...
     0.4291+1j*0.0567   0.5060-1j*0.0436  -0.5094+1j*0.0593  -0.4257-1j*0.0612 ];

CB1(:,:,6) = [ ...
     0                  0                  0                  0; ...
    -0.4042+1j*0.3244  -0.5226+1j*0.8421   0.6065-1j*0.4572   0.4600-1j*0.6814; ...
    -0.7426-1j*0.3788   0.7240+1j*0.4138  -0.4258-1j*0.6333   0.6371+1j*0.7526; ...
     0                  0                  0                  0 ];

% ----- His codebook -----
CB2(:,:,1) = [ ...
    -0.3318+1j*0.6262  -0.8304+1j*0.4252   0.8304-1j*0.4252   0.3318-1j*0.6262; ...
     0                  0                  0                  0; ...
     0.7055+1j*0.0000  -0.3601+1j*0.0000   0.3601+1j*0.0000  -0.7055+1j*0.0000; ...
     0                  0                  0                  0 ];

CB2(:,:,2) = [ ...
     0                  0                  0                  0; ...
     0.7055+1j*0.0000  -0.3601+1j*0.0000   0.3601+1j*0.0000  -0.7055+1j*0.0000; ...
     0                  0                  0                  0; ...
    -0.3318+1j*0.6262  -0.8304+1j*0.4252   0.8304-1j*0.4252   0.3318-1j*0.6262 ];

CB2(:,:,3) = [ ...
     0.3601+1j*0.0000   0.7055+1j*0.0000  -0.7055+1j*0.0000  -0.3601+1j*0.0000; ...
    -0.4202-1j*0.8350   0.5933+1j*0.3548  -0.5933-1j*0.3548   0.4202+1j*0.8350; ...
     0                  0                  0                  0; ...
     0                  0                  0                  0 ];

CB2(:,:,4) = [ ...
     0                  0                  0                  0; ...
     0                  0                  0                  0; ...
    -0.3318+1j*0.6262  -0.8304+1j*0.4252   0.8304-1j*0.4252   0.3318-1j*0.6262; ...
    -0.4202-1j*0.8350   0.5933+1j*0.3548  -0.5933-1j*0.3548   0.4202+1j*0.8350 ];

CB2(:,:,5) = [ ...
    -0.4202-1j*0.8350   0.5933+1j*0.3548  -0.5933-1j*0.3548   0.4202+1j*0.8350; ...
     0                  0                  0                  0; ...
     0                  0                  0                  0; ...
     0.3601+1j*0.0000   0.7055+1j*0.0000  -0.7055+1j*0.0000  -0.3601+1j*0.0000 ];

CB2(:,:,6) = [ ...
     0                  0                  0                  0; ...
    -0.3318+1j*0.6262  -0.8304+1j*0.4252   0.8304-1j*0.4252   0.3318-1j*0.6262; ...
    -0.4202-1j*0.8350   0.5933+1j*0.3548  -0.5933-1j*0.3548   0.4202+1j*0.8350; ...
     0                  0                  0                  0 ];

% Pack both into 4D array for convenience: (K,M,V,codebook_index)
CB_all = cat(4, CB1, CB2);
numCB  = 2;

%% -------- Basic SCMA / OFDM parameters (unchanged style) --------
[K,M,V] = size(CB1);   % should be 4,4,6

Nfft   = 1024;           % number of OFDM subcarriers
Ncp    = 32;             % cyclic prefix length
NsymTD = Nfft + Ncp;     % time-domain samples per OFDM symbol (with CP)

Q      = Nfft / K;       % SCMA blocks per OFDM symbol
assert(mod(Nfft,K)==0,'Nfft must be divisible by K');

% Bits per resource element (RE)
R = log2(M)*V/K;

% SCMA detector iterations
Niter = 6;

%% -------- AWGN-only, no fading, no CFO, no PN, no SCO --------
% Effective channel is flat and equal to 1 on all REs.
% We still go through OFDM structure, but channel = I, CFO = 0.

% Eb/N0 sweep (dB)
EbN0dB_vec = 0:1:10;      % you can extend to 46 if you like
Ns = numel(EbN0dB_vec);

%% -------- Monte-Carlo simulation control --------
NErrTarget = 1000;       % target total bit errors per (codebook, Eb/N0)
NBitsMax   = 5e7;        % safety cap on total simulated bits
Nd_batch   = 300;        % OFDM symbols per batch

% Store BER for each codebook
BER = zeros(numCB, Ns);

%% ==================== Main loops: over codebooks & Eb/N0 ====================
for icb = 1:numCB
    CB = CB_all(:,:,:,icb);   % select current codebook

    fprintf('=== Codebook %d ===\n', icb);

    for is = 1:Ns
        EbN0dB = EbN0dB_vec(is);
        SNRdB  = EbN0dB + 10*log10(R);   % SNR per RE
        SNRlin = 10^(SNRdB/10);

        % AWGN noise power per RE (frequency domain)
        N0_awgn = 1 / SNRlin;

        % No CFO / ICI here, AWGN only
        N0_eff = N0_awgn;

        % Time-domain AWGN variance (IFFT uses 1/Nfft)
        noiseVar_td = N0_awgn / Nfft;

        % Error accumulation
        err_total_vec  = zeros(1,V);    % bit errors per user
        bits_total_vec = zeros(1,V);    % total bits per user
        err_total_sum  = 0;
        bits_total_sum = 0;
        nBatch         = 0;

        while (err_total_sum < NErrTarget) && (bits_total_sum < NBitsMax)
            nBatch = nBatch + 1;

            % Number of OFDM symbols in this batch
            Nd      = Nd_batch;
            Nblocks = Nd * Q;      % number of SCMA blocks in this batch
            N       = Nblocks;

            % Random SCMA symbols for all users and blocks: 0..M-1
            x = randi([0 M-1], V, N);

            % Received REs and channel responses
            y = zeros(K, N);
            h = zeros(K, V, N);

            blk_ofs = 0;

            %% ---- Loop over Nd OFDM symbols (AWGN-only channel) ----
            for m = 1:Nd
                idx_blk = blk_ofs + (1:Q);
                x_m     = x(:, idx_blk);    % V x Q

                % 1) SCMA superposition: W (K x Q)
                W = zeros(K, Q);
                for u = 1:V
                    for q = 1:Q
                        midx = x_m(u,q) + 1;
                        W(:,q) = W(:,q) + CB(:,midx,u);
                    end
                end

                % 2) Interleaved mapping: K x Q -> Nfft
                S = zeros(Nfft,1);
                for k_re = 1:K
                    n_idx = (k_re-1)*Q + (1:Q);
                    S(n_idx) = W(k_re,:).';
                end

                % 3) IFFT + CP (baseband OFDM symbol)
                x_fd = ifft(S, Nfft);
                x_td = [x_fd(end-Ncp+1:end); x_fd];    % length NsymTD

                % 4) Flat AWGN channel: y_chan = x_td (channel gain = 1)
                y_chan = x_td;

                % 5) No CFO, no phase noise, so just:
                y_lo = y_chan;

                % 6) AWGN (time domain)
                w_td = sqrt(noiseVar_td/2) * ...
                       (randn(size(y_lo)) + 1j*randn(size(y_lo)));
                y_cp = y_lo + w_td;

                % 7) Perfect timing: remove CP from Ncp+1
                y_td_noCP = y_cp(Ncp+1 : Ncp+Nfft);

                % 8) FFT to get frequency-domain received symbol
                Y_fd = fft(y_td_noCP, Nfft);

                % 9) Flat channel frequency response H(k) = 1
                Hf = ones(Nfft,1);

                % 10) Extract K x Q resource elements
                Yk_mat = zeros(K, Q);
                Hk_mat = zeros(K, Q);
                for k_re = 1:K
                    n_idx = (k_re-1)*Q + (1:Q);
                    Yk_mat(k_re,:) = Y_fd(n_idx).';
                    Hk_mat(k_re,:) = Hf(n_idx).';
                end

                % Store received REs and flat CSI for all users / blocks
                y(:, idx_blk) = Yk_mat;
                for u = 1:V
                    h(:,u,idx_blk) = Hk_mat;  % same flat channel for all users
                end

                blk_ofs = blk_ofs + Q;
            end

            %% ---- SCMA detection (for this batch) ----
            LLR = scmadec(y, CB, h, N0_eff, Niter);

            %% ---- Bit mapping and error counting ----
            % Original transmitted bits
            r    = de2bi(x, log2(M), 'left-msb');
            data = zeros(log2(M)*N, V);
            for kk = 1:V
                data(:,kk) = reshape(downsample(r, V, kk-1).', [], 1);
            end

            % Detected bits (LLR <= 0 -> bit 1)
            datadec = reshape((LLR <= 0), [log2(M) N*V]).';
            datar   = zeros(log2(M)*N, V);
            for kk = 1:V
                datar(:,kk) = reshape(downsample(datadec, V, kk-1).', [], 1);
            end

            % Bit errors and totals per user
            err_vec  = sum(xor(data, datar));
            bits_vec = log2(M)*N * ones(1,V);

            % Accumulate
            err_total_vec  = err_total_vec  + err_vec;
            bits_total_vec = bits_total_vec + bits_vec;

            err_total_sum  = sum(err_total_vec);
            bits_total_sum = sum(bits_total_vec);
        end

        % Average BER over 6 users for this codebook / EbN0
        BER(icb, is) = mean(err_total_vec ./ bits_total_vec);

        fprintf(['CB=%d | EbN0=%2d dB | BER=%.3e, ' ...
                 'bits=%g, errors=%g, batches=%d\n'], ...
                 icb, EbN0dB, BER(icb,is), ...
                 bits_total_sum, err_total_sum, nBatch);
    end
end

%% -------- Plot both codebooks on the same figure --------
markers = {'o','s'};
colors  = lines(numCB);

figure; clf; hold on; grid on;
for icb = 1:numCB
    semilogy(EbN0dB_vec, BER(icb,:), ...
        'LineWidth', 1.6, ...
        'Marker', markers{icb}, ...
        'MarkerSize', 6, ...
        'Color', colors(icb,:), ...
        'DisplayName', sprintf('Codebook %d', icb));
end

xlabel('E_b/N_0 (dB)');
ylabel('Average BER over 6 users');
title('SCMA-OFDM over AWGN (no fading, no CFO, no impairments)');
legend('Location','southwest');
set(gca, 'YScale', 'log');
ylim([1e-6 1]);
xlim([min(EbN0dB_vec) max(EbN0dB_vec)]);

end
