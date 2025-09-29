# These are the codes required to test the generation ability for BP downstream task.

These codes requires such operations:

- Generate ECG signals from PPG signals with UniCardio
- Train blood pressure prediction tasks based on PPG, PPG with generated ECG and PPG with ECG signals

### Below are the introductions to what each code does

BP_generation_MIMICII.py: Generate ECG signals from PPG signals with UniCardio

BP_downstram_gen_Gen.py: Train blood pressure prediction model with PPG and generated ECG

BP_downstram_gen_PPG.py: Train blood pressure prediction model with PPG

BP_downstram_gen_ECG.py: Train blood pressure prediction model with PPG and ECG