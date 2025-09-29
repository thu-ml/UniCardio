# These are the codes for testing the translation and imputation abilities on the WESAD dataset

These codes requires such operations:

- Create the missing values and noises for the ECG signal
- Generate ECG signals from PPG signals with UniCardio
- Impute ECG signals with UniCardio
- Denoise ECG signals with UniCardio
- Detect heart rate on PPG, generated ECG, denoised ECG and imputed ECG


### Below are the introductions to what each code does

Imputation_generation.py: Add missing values/noises to the ECG signals and clean them with UniCardio.

ROI_generation.py: Generate ECG signals from PPG signals with UniCardio.

results_analysis.py: Detect heart rates from the PPG signals and ECG signals generated from PPG signals with UniCardio.

restoration_analysis.py: Detect heart rates from the cleaned ECG signals from UniCardio.