# Egret
Enhancing Generic Reaction Yield Prediction through Reaction Condition-Based Contrastive Learning

![Egret](./dataset/model_framework_.png)

## Contents
- [OS Requirements](#os-requirements)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Use Egret](#use-egret)

## OS Requirements
This repository has been tested on **Linux**  operating systems.

## Python Dependencies
* Python (version >= 3.7) 
* PyTorch (version >= 1.13.0) 
* RDKit (version >= 2020)
* Transformers (version == 4.18.0)
* Simpletransformers (version == 0.61.13)

## Installation Guide
Create a virtual environment to run the code of Egret.<br>
Make sure to install pytorch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/xiaodanyin/Egret.git
cd Egret
conda env create -f envs.yaml
conda activate egret_env
```
## Download
The **models** and **datasets** can be downloaded from the following link: https://drive.google.com/file/d/1K_EVByx5Vul5HuO3z_tWiVwVm37_jxPa/view?usp=sharing.

## Use Egret
You can use Egret to predict yields or  yield intervals for chemical reactions. **First** download the yield prediction model (https://drive.google.com/file/d/1MNzGyMYtNDt-TshKtsJkEEjHGtXTJhLN/view?usp=sharing) and place it under the "yield_prediction_model" folder. **Then** prepare the txt file containing the SMILES of the reaction you want to predict, and enter the following command:<br>
```
cd inference
python yield_predict.py --input_path path/to/input_file.txt \
                        --output_path path/to/output.csv \
```
For example, by using the following command, you can obtain the predicted yield values or intervals for chemical reactions in the "input_demo.txt" file.<br>
```
cd inference
python yield_predict.py --input_path test_files/input_demo.txt \
                        --output_path test_files/predicted_yields.csv \
```