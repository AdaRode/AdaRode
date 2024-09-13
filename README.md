
# AdaRode Reproduction Code Repository

This anonymous repository contains the reproduction code for the paper "Towards Cost-effective Robust Detection for Persistent Malicious Injection Variants." The code is organized to help you reproduce the results for the research questions RQ1, RQ2, and RQ3.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Reproducing RQ1](#reproducing-rq1)
- [Reproducing RQ2](#reproducing-rq2)
- [Reproducing RQ3](#reproducing-rq3)

## Requirements
- Python 3.10 or higher
- Required packages listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AdaRode/AdaRode.git
    cd AdaRode
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
Each folder contains two open-source datasets, PIK and HPD. This is the same in every required location, and the data has been stored in the project-configured location.

## Reproducing RQ1

To reproduce the results for RQ1, you need to refer to the experimental code in the `RQ1` folder.

### Experiment results on LLM (LLaMA-7B/13B)
`1Model_selection_LLM` stores the results for LLaMA-7B/13B. You can simply run `result_analysis.py` to replicate my experimental results. The analysis target can be selected between lines 5-15 in the script.

```bash
python result_analysis.py
```
To replicate the training and testing results of the LLM(LLama-7B/13B), follow these steps:

1. **Navigate to the appropriate folder:**
   - Each model and dataset combination has its specific folder.

2. **Run the Training Script:**
   - Execute the training script by running:
     ```bash
     python Train_LLM.py
     ```

3. **Configure and Run the Testing Script:**
   - Open `Test_LLM.py`.
   - Set the best-trained model on lines 58-59.
   - Run the testing script by executing:
     ```bash
     python Test_LLM.py
     ```
### Experiment Results on XLNet, RoBERTa, BERT, and T5

To replicate the experiment results for XLNet, RoBERTa, BERT, and T5 models, follow these steps:

1. **Download the Base Models from hugging face:**
   - Download the base versions of each model and place them in the appropriate folders.

2. **Configure Paths:**
   - Set the model paths in `Config/Train.yaml` and `Config/Test.yaml`.

3. **Running Tests:**
   - You can simply run the test script:
     ```bash
     python Test.py
     ```
   - The model parameters are obtained from the `Model` directory. Ensure the base model paths are correctly configured to replicate the results.

4. **Re-training the Models:**
   - If you need to retrain the models:
     1. Configure your training parameters in `Config/Train.yaml`.
     2. Run the training script:
        ```bash
        python Train.py
        ```
### Experiment Results on Attack Parameter Selection

To reproduce the results, you can run the scripts `acc_analysis.py` and `Iter_analysis.py` located in the `Result/` directory.

#### Steps to Reproduce attack process:

1. **Obtain Parameters:**
   - Retrieve appropriate parameters from the `Model/` directory.

2. **Re-run Attack Process:**
   - Modify the parameters in the `Config/adv_config.yaml` file as needed.
   - Execute the script `Aug/AdaRode.py` to initiate the attack process:

```bash
python Aug/AdaRode.py
```


## Reproducing RQ2
To reproduce the results for RQ2, run the following command:
# README.md

## Overview
This project aims to evaluate the performance of various detectors on original attacks and attacks using MIVs. Below are the instructions to reproduce the results and retrain the models.

## Baseline Reproducibility

### Reproducing Detector Performance on Original Attack
1. Navigate to the `Baseline_Reproducibility` directory.
2. Download the pre-trained model for each detector.
3. Run `test.py` to reproduce the results.

### Retraining the Model
To retrain the model, execute the following script:
```bash
python train.py
```

## Reproducing Attack using MIV

### Steps to Reproduce
1. Download the pre-trained models or use your previously trained models.
2. Modify the attack model in `@RSAttacker/Attackers/Rsample.py` by editing lines 19-56.
3. Run the attack script to generate a new data file. This file will be saved in `@RSAttacker/Attackers/AdvMut_data/`.

### Evaluating Attack using MIV
1. Move the generated data file to the `Baseline_Reproducibility` directory.
2. Modify `test.py` to switch between evaluating original attack data and attack using MIV data:
```python
# Random Attack data
# Texts = data['adv_raw']
# Labels = data['adv_label']

# Origin data for check model
Texts = data['ori_raw']
Labels = data['adv_label']
```
3. Run `test.py` to evaluate the results.

## Device Configuration
Ensure the correct device configuration in `test.py`:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


### Test Our AdaRode

1. **Test without PMIV**:
   ```bash
   python \AdaRode-A&N\Test_withoutPMIV.py
   ```

2. **Test with PMIV**:
   ```bash
   python \AdaRode-A&N\Test_withPMIV.py
   ```

### Conducting Real-time Attack

1. Insert your model into the `@RSAttacker` module to perform the attack and obtain real-time data.

2. Place the obtained data into the `\AdaRode-A&N\AttackRS_data` directory.

3. Run the scripts again to obtain the final results.


## Reproducing RQ3

To reproduce the results for RQ3, run the following command:

```bash
python \RQ3\Result\HPD\statis_result.py
python \RQ3\Result\PIK\statis_result.py
```

To regenerate the entire attack process, execute each algorithm under RQ3 by running:

```bash
python RQ3/Aug/<algorithm_name>_<dataset_name>.py
```

Retrieve the JSON data from each algorithm's `augdata` and store it in the corresponding location within `Result` to replicate the entire attack process.

## Reproducing Discussions

### Running Test Scripts to Reproduce Results

To reproduce the results, you can simply run the `test.py` scripts located in the following directories:

```plaintext
Disscussion\Disscussion\@FSE_Disscussion_2_TestTableIX\DBN\test.py
Disscussion\Disscussion\@FSE_Disscussion_2_TestTableIX\RNN\test.py
```

### Reproducing the Enhancement Process

To fully reproduce the enhancement process, follow these steps:

1. **Attack Detection Models using AdaRode:**

   Navigate to the `Disscussion\Disscussion\` directory and run the attack scripts for the detection models.

   ```plaintext
   Disscussion\Disscussion\<Algorithm_Folder>\AdaRode\Aug\AdaRode_PIK.py
   ```

2. **Generate Adversarial Samples:**

   This will generate adversarial samples and save them in the following file:

   ```plaintext
   AdaRode\augdata\adv_data_PIK.pickle
   ```

3. **Rename and Relocate Adversarial Samples:**

   Rename the adversarial sample file and move it to the corresponding directory for retraining.

   ```plaintext
   <Algorithm_Folder>\AdaRetrain\Data\adv_data.pickle
   ```

4. **Retrain Models with Adversarial Data:**

   Use the renamed adversarial samples for adversarial training to obtain enhanced models.

5. **Test Enhanced Models:**

   Place the retrained models in the appropriate directory for testing.

   ```plaintext
   Disscussion\Disscussion\@FSE_Disscussion_2_TestTableIX\<Model_Folder>\
   ```

Now, run the `test.py` scripts again to evaluate the enhanced models.

