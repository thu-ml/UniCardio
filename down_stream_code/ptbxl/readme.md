# These are the codes required to test the imputation and denoising ability on the ptbxl dataset.

These codes requires such operations:

- Processing the ptbxl data
- Create signals with noise or missing values, then fix them with UniCardio
- Perform classification on certain diseases based on the clean, dirty and cleaned signals.

### Below are the introductions to what each code does

Processing.py: used to clean the ptbxl data

generation.py: Manually add noise or missing values to the clean ptbxl data, and generate the cleaned version

classification_denoise_final.py: run classification on the clean, noisy and denoised signals.

classification_imputation_final.py: run classification on the clean, noisy and imputed signals.

For different diseases, simply change the label index in the two classification codes according to the ptbxl data's label.