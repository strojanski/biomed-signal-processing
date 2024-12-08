import wfdb
from scipy.signal import butter, lfilter

def read_record(record_name):
    record = wfdb.rdrecord(record_name)
    return record

def read_annotation(record_name):
    annotation = wfdb.rdann(record_name, 'atr')
    return annotation
import numpy as np

def create_haar_like_filter(fs, B1=None, B2=None):
    """
    Creates a Haar-like matched filter for ECG QRS detection.
    """
    B1 = int(0.025 * fs)
    B2 = int(0.06 * fs)
    c = 2 * (B2 - B1) / (2 * B1 + 1)
    h = np.zeros(2 * B2 + 1)
    h[:B1] = c
    h[B1:B2] = -1
    return h


def calculate_convolution_delay(kernel, fs):
    """
    Calculate the delay introduced by convolution.

    Parameters:
    - kernel: The convolution filter kernel (1D array).
    - fs: Sampling frequency of the signal (Hz).

    Returns:
    - delay_samples: Delay in samples.
    - delay_time: Delay in seconds.
    """
    delay_samples = (len(kernel) - 1) // 2
    delay_time = delay_samples / fs
    return delay_samples, delay_time

def apply_matched_filter(signal, fs, B1=None, B2=None):
    """
    Applies the Haar-like matched filter to an ECG signal.
    """
    h = create_haar_like_filter(fs, B1, B2)
    
    filtered_signal = np.convolve(signal, h, mode='same')
    return filtered_signal

def second_order_difference(signal):
    """
    Computes the second-order difference of the signal.
    """
    x2 = 2 * signal - np.roll(signal, 1) - np.roll(signal, -1)
    x2[0] = 0  # Handle edge cases
    x2[-1] = 0
    return x2

def calculate_score(signal, filtered_signal, second_diff, c1=0.55):
    """
    Calculates the score function as described in the paper.
    """
    
    s = np.array([filtered_signal[n]*(signal[n] + c1*second_diff[n]) for n in range(len(filtered_signal))])
    return s

def sift_r_wave_candidates(score, fs, threshold=0.1):
    """
    Sifts the R-wave peak candidates from the ECG signal.
    """
    # Adaptive thresholding to identify R-wave candidates
    candidates = []
    sampling_interval = int(0.2 * fs)  # Minimum interval between candidates
    for k in range(len(score)):
        start = max(0, k - sampling_interval)
        end = min(len(score), k + sampling_interval + 1)
    
        is_max = True
                    
        for n in range(start, end):
            if n == k:
                continue
            if abs(score[n]) > abs(score[k]):
                is_max = False
                break
        if is_max and abs(score[k]) > threshold:
            candidates.append(k)
    
    return candidates

def sift_r_wave_candidates_dynamic(score, fs, percentile=80, min_value=0.1):
    """
    Sifts the R-wave peak candidates from the ECG signal with adaptive thresholding.

    Parameters:
    - score: The score function values s[n] from the ECG signal.
    - fs: Sampling frequency of the signal.
    - percentile: The percentile (e.g., 90th) of the score values to set the dynamic threshold.
    - window_size: The size of the window to calculate the local threshold.

    Returns:
    - candidates: Indices of the R-wave peak candidates after sifting.
    """
    # Calculate the dynamic threshold based on the percentile of the score values
    threshold = max(np.percentile(np.abs(score), percentile), min_value)
    
    candidates = []
    sampling_interval = int(0.2 * fs)  # Minimum interval between candidates

    # Iterate through all score values to identify local maxima
    for k in range(len(score)):
        start = max(0, k - sampling_interval)
        end = min(len(score), k + sampling_interval + 1)
    
        is_max = True
        # Check if score[k] is a local maximum
        for n in range(start, end):
            if n == k:
                continue
            if abs(score[n]) > abs(score[k]):
                is_max = False
                break
        
        # If it's a local maximum and the score is above the dynamic threshold, add it to candidates
        if is_max and abs(score[k]) > threshold:
            candidates.append(k)
    
    return candidates


import numpy as np

def calculate_adaptive_threshold(score_values, r_peak_candidates, fs, recent_window=10, T=0.5, beta1=0.1, beta2=0.1):
    """
    Calculates the adaptive threshold based on score values and RR interval regularity.

    Parameters:
    - score_values: The score function values s[n] from the ECG signal.
    - r_peak_candidates: Indices of sifted R-wave peak candidates.
    - fs: Sampling frequency of the signal (Hz).
    - recent_window: Time window (seconds) for considering recent peaks (default 10).
    - T: Constant for threshold adjustment.
    - beta1, beta2: Constants for regularity-based threshold adjustment.

    Returns:
    - adaptive_threshold: The adaptive threshold for R-wave detection.
    """
    # Get recent candidates within the last `recent_window` seconds
    recent_window_samples = recent_window * fs
    recent_candidates = [c for c in r_peak_candidates if c > len(score_values) - recent_window_samples]

    if len(recent_candidates) < 2:
        # If insufficient candidates, return a fallback threshold
        return T

    # Calculate S5 (5th largest score value or median if fewer candidates)
    sorted_scores = np.sort(np.abs(score_values[recent_candidates]))
    S5 = sorted_scores[-5] if len(sorted_scores) >= 5 else np.median(sorted_scores)

    # Calculate W1
    W1 = T + S5

    # Calculate RR intervals
    rr_intervals = np.diff(recent_candidates)
    if len(rr_intervals) < 1:
        return W1  # Fallback to W1 if no intervals are available

    # Dynamically adjust weights to match the number of RR intervals
    weights = np.linspace(1, len(rr_intervals), len(rr_intervals))
    weights = weights / weights.sum()  # Normalize weights
    Ie = np.dot(weights, rr_intervals)

    # Regularity adjustment
    W2 = beta1 + beta2 * (np.abs(rr_intervals[-1] - Ie) if len(rr_intervals) > 1 else 0)

    # Final adaptive threshold
    adaptive_threshold = W1 * W2
    return adaptive_threshold



def refine_r_wave_candidates(score_values, r_peak_candidates, fs, threshold_func):
    """
    Refines R-wave candidates using an adaptive threshold.

    Parameters:
    - score_values: The score function values s[n].
    - r_peak_candidates: Indices of sifted R-wave peak candidates.
    - fs: Sampling frequency of the signal (Hz).
    - threshold_func: Function to calculate the adaptive threshold.

    Returns:
    - refined_candidates: Indices of refined R-wave peaks.
    """
    if not r_peak_candidates:
        return np.array([])  # Return empty array if no candidates

    adaptive_threshold = threshold_func(score_values, r_peak_candidates, fs)
    print(f"Adaptive Threshold: {adaptive_threshold}")

    refined_candidates = [
        candidate for candidate in r_peak_candidates
        if np.abs(score_values[candidate]) > adaptive_threshold
    ]
    
    return np.array(refined_candidates)


import matplotlib.pyplot as plt

def plot_detected_peaks(signal, peak_indices, fs, ann=None, title="R-Wave Peak Detection"):
    """
    Plots the ECG signal and marks the detected R-wave peaks.

    Parameters:
    - signal: The ECG signal (1D array).
    - peak_indices: Indices of detected R-wave peaks (list or array).
    - fs: Sampling frequency of the signal (Hz).
    - ann: Optional array of annotation indices to compare with detected peaks.
    - title: Title of the plot (string).
    """
    # Create time axis for the signal
    time_axis = np.arange(len(signal))

    # Check if peak_indices are valid
    peak_indices = np.asarray(peak_indices)
    peak_indices = peak_indices[peak_indices < len(signal)]

    # Plot the ECG signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, signal, label="ECG Signal", linewidth=1.5, alpha=0.8)

    # Highlight the detected R-wave peaks
    if len(peak_indices) > 0:
        plt.scatter(
            time_axis[peak_indices], 
            signal[peak_indices], 
            color="red", 
            label="Detected Peaks", 
            marker="o", 
            s=50
        )

    # Highlight the annotation peaks if provided
    if ann is not None:
        ann = np.asarray(ann)
        ann = ann[ann < len(signal)]  # Ensure annotations fit within signal length
        plt.scatter(
            time_axis[ann],
            signal[ann],
            color="green",
            label="Annotation Peaks",
            marker="x",
            s=50
        )

    # Add labels and legend
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


import numpy as np

def variation_ratio_test(signal, peak_candidates, fs, window=0.1, threshold=0.5):
    """
    Applies the Variation Ratio Test to refine R-wave peak detection.

    Parameters:
    - signal: The baseline-corrected ECG signal (1D array).
    - peak_candidates: List or array of indices of R-wave peak candidates.
    - fs: Sampling frequency of the ECG signal (Hz).
    - window: The time window (seconds) to evaluate the region around each peak (default: 0.1).
    - threshold: The minimum ratio (Ω) to retain a peak as a true R-wave peak.

    Returns:
    - refined_peaks: List of indices for R-wave peaks passing the variation ratio test.
    """
    
    # Dynamically adjust threshold based on signal noise
    signal_noise_level = np.std(signal)
    if signal_noise_level > 0.2:  # High noise
        threshold *= 0.4
    elif signal_noise_level < 0.1:  # Low noise
        threshold *= 0.6
    else:  # Moderate noise
        threshold *= 0.5

    refined_peaks = []
    window_samples = int(window * fs)

    for m in peak_candidates:
        # Define the region around the candidate peak
        region_start = max(0, m - window_samples)
        region_end = min(len(signal), m + window_samples + 1)
        region = signal[region_start:region_end]

        # Calculate mean amplitude for normalization
        mean_amplitude = np.mean(np.abs(region)) if np.mean(np.abs(region)) != 0 else 1

        # Calculate u1 (peak-to-peak variation), normalized
        u1 = (np.max(region) - np.min(region)) / mean_amplitude

        # Calculate u2 (total variation), normalized
        u2 = np.sum(np.abs(region[1:] - region[:-1])) / mean_amplitude

        # Calculate variation ratio Ω
        omega = u1 / u2 if u2 > 0 else 0

        # Retain the peak if Ω > threshold
        if omega > threshold:
            refined_peaks.append(m)

    return refined_peaks



def apply_high_pass_filter(signal, cutoff_freq, fs, order=4):
    """
    Applies a high-pass Butterworth filter to the signal.
    
    Parameters:
    - signal: Input ECG signal (1D array).
    - cutoff_freq: Cutoff frequency of the filter in Hz.
    - fs: Sampling frequency of the signal in Hz.
    - order: Order of the Butterworth filter (default is 4).
    
    Returns:
    - filtered_signal: Signal after high-pass filtering.
    """
    # Normalize the cutoff frequency to the Nyquist frequency
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist

    # Design the Butterworth high-pass filter
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)

    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


def sensitivity(tp, fn):
    """
    Calculate the Sensitivity (SE) metric.
    
    SE = TP / (TP + FN)
    """
    return (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0  # Avoid division by zero


def positive_prediction(tp, fp):
    """
    Calculate the Positive Prediction (+P) metric.
    
    +P = TP / (TP + FP)
    """
    return (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0  # Avoid division by zero


def detection_error_rate(tp, fn, fp):
    """
    Calculate the Detection Error Rate (DER) metric.
    
    DER = (FP + FN) / (TP + FN)
    """
    return ((fp + fn) / (tp + fn)) * 100 if (tp + fn) > 0 else 0  # Avoid division by zero

def get_metrics(labels, detections, offset_threshold):
    """
    Calculate the performance metrics (Sensitivity, Positive Prediction, and Detection Error Rate)
    by comparing the detected R-wave peaks with the ground truth labels.
    
    Parameters:
    - labels: Ground truth R-wave peak locations.
    - detections: Detected R-wave peak locations.
    - offset_threshold: The maximum allowable offset (in samples) for a detection to be considered a match to a label.

    Returns:
    - tp: Number of True Positives.
    - fn: Number of False Negatives.
    - fp: Number of False Positives.
    """
    tp = 0  # True positives
    fn = 0  # False negatives
    fp = 0  # False positives

    unmatched_detections = detections.copy()  # Copy of detections to remove matched ones

    # Calculate True Positives (TP) and False Negatives (FN)
    for label in labels:
        matched = False
        for detection in detections:
            if abs(label - detection) <= offset_threshold:  # Considered a match if within the offset threshold
                tp += 1
                matched = True
                if detection in unmatched_detections:
                    unmatched_detections.remove(detection)  # Remove matched detection from the list
                break
        if not matched:
            fn += 1  # If no match is found, it's a false negative

    # Calculate False Positives (FP)
    for detection in unmatched_detections:
        fp += 1  # Any unmatched detection is a false positive

    # Calculate Sensitivity (SE), Positive Prediction (+P), and Detection Error Rate (DER)
    sensitivity_value = sensitivity(tp, fn)
    positive_prediction_value = positive_prediction(tp, fp)
    detection_error_rate_value = detection_error_rate(tp, fn, fp)

    print(f"Sensitivity (SE): {sensitivity_value:.2f}%")
    print(f"Positive Prediction (+P): {positive_prediction_value:.2f}%")
    print(f"Detection Error Rate (DER): {detection_error_rate_value:.3f}%")

    return f"Sensitivity (SE): {sensitivity_value:.2f}%\nPositive Prediction (+P): {positive_prediction_value:.2f}%\nDetection Error Rate (DER): {detection_error_rate_value:.3f}%\n"


def get_norm_metrics(labels, detections, offset_threshold, n_labels):
    """
    Calculate the performance metrics (Sensitivity, Positive Prediction, and Detection Error Rate)
    by comparing the detected R-wave peaks with the ground truth labels.
    
    Parameters:
    - labels: Ground truth R-wave peak locations.
    - detections: Detected R-wave peak locations.
    - offset_threshold: The maximum allowable offset (in samples) for a detection to be considered a match to a label.

    Returns:
    - tp: Number of True Positives.
    - fn: Number of False Negatives.
    - fp: Number of False Positives.
    """
    tp = 0  # True positives
    fn = 0  # False negatives
    fp = 0  # False positives

    unmatched_detections = detections.copy()  # Copy of detections to remove matched ones

    # Calculate True Positives (TP) and False Negatives (FN)
    for label in labels:
        matched = False
        for detection in detections:
            if abs(label - detection) <= offset_threshold:  # Considered a match if within the offset threshold
                tp += 1
                matched = True
                if detection in unmatched_detections:
                    unmatched_detections.remove(detection)  # Remove matched detection from the list
                break
        if not matched:
            fn += 1  # If no match is found, it's a false negative

    # Calculate False Positives (FP)
    for detection in unmatched_detections:
        fp += 1  # Any unmatched detection is a false positive

    # Calculate Sensitivity (SE), Positive Prediction (+P), and Detection Error Rate (DER)
    sensitivity_value = sensitivity(tp, fn) * n_labels
    positive_prediction_value = positive_prediction(tp, fp) * n_labels
    detection_error_rate_value = detection_error_rate(tp, fn, fp) * n_labels

    print(f"Sensitivity (SE): {sensitivity_value:.2f}%")
    print(f"Positive Prediction (+P): {positive_prediction_value:.2f}%")
    print(f"Detection Error Rate (DER): {detection_error_rate_value:.3f}%")

    return f"Sensitivity (SE): {sensitivity_value:.2f}%\nPositive Prediction (+P): {positive_prediction_value:.2f}%\nDetection Error Rate (DER): {detection_error_rate_value:.3f}%\n"
