
# AdaRode Reproduction Code Repository

This repository contains the reproduction code for the paper "Towards Cost-effective Robust Detection for Persistent Malicious Injection Variants." The code is organized to help you reproduce the results for the research questions RQ1, RQ2, and RQ3.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Reproducing RQ1](#reproducing-rq1)
- [Reproducing RQ2](#reproducing-rq2)
- [Reproducing RQ3](#reproducing-rq3)
- [Contact](#contact)

## Requirements
- Python 3.10 or higher
- Required packages listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AdaRode.git
    cd AdaRode
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
Each folder contains two open-source datasets PIK & HPD.

## Reproducing RQ1
To reproduce the results for RQ1, you need to refer to the experimental code in the RQ1 folder.







## Reproducing RQ2
To reproduce the results for RQ2, run the following command:
```bash
python reproduce_rq2.py --data_dir ./data --output_dir ./results/rq2
```
This script will execute the experiments and save the results in the specified output directory.

## Reproducing RQ3
To reproduce the results for RQ3, run the following command:
```bash
python reproduce_rq3.py --data_dir ./data --output_dir ./results/rq3
```
This script will execute the experiments and save the results in the specified output directory.

## Contact
If you have any questions or run into any issues, please open an issue on this repository or contact us at [your email address].
