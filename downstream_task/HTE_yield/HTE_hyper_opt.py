import os
import pandas as pd
from rxnfp.models import SmilesClassificationModel
import sys
sys.path.append('../../preprocess_script/benchmark_data')
from Buchwald_Hartwig_data import generate_buchwald_hartwig_rxns
import sklearn

try:
    import wandb
    wandb_available = True
except ImportError:
    raise ValueError('Wandb is not available')

def main():
    def train():
        wandb.init()
        print("HyperParams=>>", wandb.config)
        model_args = {
        'wandb_project': "doyle_random_01",
        'num_train_epochs': 30,
        'overwrite_output_dir': True,
        'gradient_accumulation_steps': 1, 
        "warmup_ratio": 0.00,
        "train_batch_size": 16, 
        'regression': True, 
        "num_labels":1,
        "fp16": False, 
        "evaluate_during_training": True,
        "max_seq_length": 300,  
        "config" : {'hidden_dropout_prob': wandb.config.dropout_rate},
        'learning_rate': wandb.config.learning_rate,
        }
        model_path =  os.path.join('../../outputs/egret_pretrain_stage_2/bestoutput')
        model = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=True, cuda_device=1, ignore_mismatched_sizes=True)
        model.train_model(train_df, eval_df=val_df, r2=sklearn.metrics.r2_score)

    df_doyle = pd.read_excel('../../dataset/source_dataset/Buchwald_Hartwig_reaction/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01')
    df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle)
    train_df = df_doyle.iloc[:2373][['rxn', 'Output']]
    val_df = df_doyle.iloc[2373:2768][['rxn', 'Output']]
    train_df.columns = ['text', 'labels']
    val_df.columns = ['text', 'labels']
    mean = train_df.labels.mean()
    std = train_df.labels.std()
    train_df['labels'] = (train_df['labels'] - mean) / std
    val_df['labels'] = (val_df['labels'] - mean) / std

    sweep_config = {
        'method': 'bayes', # grid, random, bayes
        'metric': {
          'name': 'r2',
          'goal': 'maximize'   
        },
        'parameters': {  
            'learning_rate': {
                'min': 0.0001,
                'max': 0.01  
            },
            'dropout_rate': {
                'min': 0.001,
                'max': 0.5,
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="doyle_random_01_hyperparams_sweep")
    wandb.agent(sweep_id, function=train, count=100)

if __name__ == '__main__':
    main()