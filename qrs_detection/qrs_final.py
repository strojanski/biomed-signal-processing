import matplotlib.pyplot as plt
import numpy as np
import os
import wfdb
from utils import *



data_dir = "data/mit-bih-arrhythmia-database-1.0.0/"
records = os.listdir(data_dir)
records = sorted(list(set([record.split(".")[0] for record in records if record.endswith(".dat")])))

symbols = ["N"]
ress = []
for rec_name in records:
    rec = read_record(f"{data_dir}{rec_name}")

    # Read data
    sig = rec.p_signal
    ann = read_annotation(f"{data_dir}{rec_name}")

    # Apply high-pass filter
    orig_sig = sig.copy()   
    sig = apply_high_pass_filter(sig, 0.6, rec.fs, order=3)

    sample_len = len(sig)

    fs = rec.fs  # Sampling frequency
    signal = sig[:sample_len, 0]  # Example signal

    # Step 1: Apply the Haar-like matched filter
    fil = create_haar_like_filter(fs)
    delay_smp, delay_time = calculate_convolution_delay(fil, fs)

    print("Signal delayed by", delay_smp, "samples.")

    filtered_signal = apply_matched_filter(signal, fs)
    # filtered_signal = filtered_signal / np.max(filtered_signal)                 # Normalize the filtered signal
    filtered_signal = [0 for i in range(delay_smp)] + list(filtered_signal)     # Add delay to the filtered signal
    filtered_signal = filtered_signal[:sample_len]                                    # Truncate the filtered signal
    filtered_signal = np.array(filtered_signal)

    # Step 2: Calculate the second-order difference
    second_diff = second_order_difference(signal)

    # Step 3: Calculate the score function
    score = calculate_score(second_diff, filtered_signal, second_diff)

    # Step 4: Sift R-wave candidates
    # r_wave_candidates = sift_r_wave_candidates(score, fs, threshold=np.max(score)/150)
    r_wave_candidates = sift_r_wave_candidates_dynamic(score, fs, percentile=80, min_value=np.max(score)/50)

    # Output results
    print("Filtered Signal:", filtered_signal)
    print("Second-Order Difference:", second_diff)
    print("Score Function:", score)
    print("R-wave Peak Candidates (indices):", r_wave_candidates)
    print(max(score))

    print("Found {} R-wave candidates".format(len(r_wave_candidates)))

    # Step 3: Find refined r wave candidates
# Compute the adaptive threshold and refine candidates
    refined_r_wave_candidates = refine_r_wave_candidates(
        score, r_wave_candidates, fs, 
        lambda score, peaks, fs: calculate_adaptive_threshold(score, peaks, fs, 0.01, 1e-5, 1e-5)
    )


    # Perform Variation Ratio Test
    refined_peaks_variation_test = variation_ratio_test(
        signal=signal, 
        peak_candidates=refined_r_wave_candidates, 
        fs=fs, 
        window=0.15,
        threshold=0.25
    )
    
    symbols = np.array(["N" for i in range(len(refined_peaks_variation_test))])
    
    wfdb.io.wrann(rec_name, "qrs", np.array(refined_peaks_variation_test), symbol=symbols)

    labels = ann.sample[1:]
    print(rec_name)
    res = get_metrics(labels, refined_peaks_variation_test, 10)
    ress.append(res)


