import os
import numpy as np
from utils import *
from scipy.signal import butter, lfilter
import wfdb
import matplotlib.pyplot as plt

# Define Paths
data_dir = "data/mit-bih-arrhythmia-database-1.0.0/"

# Utility Functions
def get_records(data_dir):
    files = os.listdir(data_dir)
    return sorted(list(set([file.split(".")[0] for file in files if file.endswith(".dat")])))

def safe_normalize(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def check_signal_dimensions(signal):
    return signal[:, 0] if len(signal.shape) > 1 else signal

# Main Processing
records = get_records(data_dir)
results = []

for rec_name in records:
    print(f"Processing record: {rec_name}")
    rec = read_record(os.path.join(data_dir, rec_name))
    ann = read_annotation(os.path.join(data_dir, rec_name))

    # Process Signal
    signal = check_signal_dimensions(rec.p_signal)
    signal = safe_normalize(signal)
    fs = rec.fs

    # High-pass Filter
    filtered_signal = apply_high_pass_filter(signal, cutoff_freq=0.6, fs=fs, order=3)

    # Haar-like Matched Filter
    matched_filter = create_haar_like_filter(fs)
    filtered_signal = apply_matched_filter(filtered_signal, fs)
    delay_smp, _ = calculate_convolution_delay(matched_filter, fs)
    filtered_signal = np.pad(filtered_signal, (delay_smp, 0), mode='constant')[:len(signal)]

    # Second-order Difference
    second_diff = second_order_difference(signal)

    # Score Calculation
    score = calculate_score(signal, filtered_signal, second_diff)

    # R-wave Candidate Detection
    r_wave_candidates = sift_r_wave_candidates_dynamic(score, fs, percentile=90, min_value=np.max(score) / 100)

    # Adaptive Threshold Refinement
    refined_r_wave_candidates = refine_r_wave_candidates(
        score, r_wave_candidates, fs,
        lambda s, peaks, f: calculate_adaptive_threshold(s, peaks, f, T=0.5)
    )

    # Variation Ratio Test
    final_peaks = variation_ratio_test(
        signal=signal,
        peak_candidates=refined_r_wave_candidates,
        fs=fs,
        window=0.15,
        threshold=0.3
    )

    # Metrics Calculation
    labels = ann.sample[1:] if rec_name == "100" else []
    metrics = get_metrics(labels, final_peaks, offset_threshold=10)
    results.append((rec_name, metrics))

    print(metrics)

# Save Results
with open("results.txt", "w") as f:
    for rec_name, metric in results:
        f.write(f"Record: {rec_name}\n{metric}\n")
