import argparse
import pandas as pd
import sys
import os
sys.path.append('../../preprocess_script/benchmark_data')
from Buchwald_Hartwig_data import generate_buchwald_hartwig_rxns
from rxnfp.models import SmilesClassificationModel 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_epochs', default=30, help="training epochs", type=int)
    parser.add_argument('--learning_rate', 
                        default=0.0008766577347233683, 
                        help="learning rate", 
                        type=float)
    parser.add_argument('--hidden_dropout_prob', 
                        default=0.00950956259888493, 
                        help="dropout rate", 
                        type=float)
    parser.add_argument('--model_path', 
                        default='../../outputs/egret_pretrain_stage_2/bestoutput', 
                        help="model path", 
                        type=str)    
    return parser.parse_args()

if __name__ == "__main__":
    debug = True
    args = get_args()   

    OUT_OF_SAMPLE_SPLIT = [
    ('Test1', 3058),
    ('Test2', 3056),
    ('Test3', 3059),
    ('Test4', 3056),
    ]    
    for sheet_name, split_point in OUT_OF_SAMPLE_SPLIT:
        print(f'#############{sheet_name}##############')
        for seed in [42, 69, 2222, 2626]:
            df = pd.read_excel('../../dataset/source_dataset/Buchwald_Hartwig_reaction/Dreher_and_Doyle_input_data.xlsx', sheet_name=sheet_name)
            df['rxn'] = generate_buchwald_hartwig_rxns(df)
            train_df = df.iloc[:split_point-1][['rxn', 'Output']] 
            test_df = df.iloc[split_point-1:][['rxn', 'Output']]
            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']
            mean = train_df.labels.mean()
            std = train_df.labels.std()
            train_df['labels'] = (train_df['labels'] - mean) / std
            test_df['labels'] = (test_df['labels'] - mean) / std
            
            model_args = {
            'num_train_epochs': args.num_train_epochs, 
            'overwrite_output_dir': True,
            'learning_rate': args.learning_rate,
            'gradient_accumulation_steps': 1,
            'regression': True, 
            "num_labels":1, 
            "fp16": False,
            "evaluate_during_training": True, 
            'manual_seed': seed,
            "output_dir": f"../../outputs/buchwald_hartwig/{sheet_name}_split_{split_point}_{seed}",
            "max_seq_length": 300, 
            "train_batch_size": 16,
            "warmup_ratio": 0.00,
            "config" : { 'hidden_dropout_prob': args.hidden_dropout_prob } ,  
            'wandb_project': False,
            "use_multiprocessing_for_evaluation":False,
            "use_multiprocessing":False,
            }
            model_path =  args.model_path
            model = SmilesClassificationModel("bert", 
                                            model_path, 
                                            num_labels=1, 
                                            args=model_args, 
                                            use_cuda=True if not debug else False, 
                                            cuda_device=1 , 
                                            ignore_mismatched_sizes=True)
            model.train_model(train_df, eval_df=test_df)
            print()    

