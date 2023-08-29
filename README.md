# Identification of Molecules in Line Forests Through Neural Network

## Scripts for the ASIAA summer project

### Description

We use a Convolutional Neural Network to automatically identify different molecules present in a given spectrum. The script be can run easily, and it outputs information with tables and graphs.

### Requirements
- Python 3.7 or higher
- Python packages:
    - numpy
    - pandas
    - matplotlib
    - scipy
    - tqdm
    - tabulate
    - re
    - torch
    - torchsummary

### Usage

1. Set up python3 and install the required python packages
1. Clone this repository
1. Place the tsv/txt spectrum file in the `Spectra` directory
1. Modify the required (and some optional) parameters for each python script by editting the `./run_spec_analysis.sh` file
    - Use commands like `python ./gauss_fit.py --help` to check descriptions of parameters in each python script
1. `cd` to the base directory and run `./run_spec_analysis.sh` to analyze the spectrum

- `profile_generator_parallel/`: generate training data
- `profile_generator_multispecies/`: generate mock spectra for testing
- `CNN_train.ipynb`: notebook for training CNN using GPU

## Contact

- Chih-Fu Lai: lycoris1062@gmail.com


