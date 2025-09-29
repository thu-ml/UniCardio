# These are the codes for testing the PPG/ECG to BP abilities on the MIMIC dataset

These codes requires such operations:

- The UniCardio code for finetuning
- Test the code


### Below are the introductions to what each code does

diffusion_model_no_compress_finetune.py: The updated UniCardio code for finetuning. The example finetune function is finetuning ECG to BP. The specific task can be changed by changing the observed data, noisy data and attention mask, just like UniCardio.

train_finetune.py: Train the finetuning task.

test_fine: Test the finetuned task.