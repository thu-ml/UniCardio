# These are the codes for testing the translation and imputation abilities on the AF dataset

These codes requires such operations:

- Create the missing values for the ECG signal
- Generate ECG signals from PPG signals with UniCardio
- Impute ECG signals with UniCardio
- Detect AF detection on PPG, generated ECG, imputed ECG and ground truth ECG


### Below are the introductions to what each code does

preprocessing_AF.py: Clean and prepare the data from AF dataset.

AF_imputation_generation.py: Add missing values to the ECG signals and clean them with UniCardio.

AF_generation.py: Generate ECG signals from PPG signals with UniCardio.

final_AF_10_trials.py: Detect AF from the PPG signals, ECG signals and ECG signals generated from PPG signals with UniCardio. By changing which signal to use in the VGG model, the train and test will be the truth ECG, generated ECG and PPG signals.

AF_imp.py: Detect AF from the ECG signals and ECG signals imputed by UniCardio. By changing which signal to use in the VGG model, the train and test will be the truth ECG and the imputed ECG signals..