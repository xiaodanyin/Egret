import argparse
import pandas as pd
from rxnfp.models import SmilesClassificationModel 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_epochs', default=160, help="training epochs", type=int)
    parser.add_argument('--learning_rate', 
                        default=0.00015754353058490168, 
                        help="learning rate", 
                        type=float)
    parser.add_argument('--hidden_dropout_prob', 
                        default=0.014763683260671429, 
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

    NAME_SPLIT = [
    ('random_split_0', 4032),
    ('random_split_1', 4032),
    ('random_split_2', 4032),
    ('random_split_3', 4032),
    ('random_split_4', 4032),
    ('random_split_5', 4032),
    ('random_split_6', 4032),
    ('random_split_7', 4032),
    ('random_split_8', 4032),
    ('random_split_9', 4032),
    ]

    for name, split in NAME_SPLIT:
        print(f'#############{name}##############')
        df = pd.read_csv(f'../../dataset/source_dataset/Suzuki_Miyaura_reaction/random_splits/{name}.tsv', sep='\t')
        train_df = df.iloc[:split][['rxn', 'y']] 
        test_df = df.iloc[split:][['rxn', 'y']]
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
        'manual_seed': 42,
        "output_dir": f"../../outputs/suzuki_miyaura/{name}_split_{split}",
        "max_seq_length": 300, 
        "train_batch_size": 16,
        "warmup_ratio": 0.00,
        "config" : { 'hidden_dropout_prob': args.hidden_dropout_prob } 
        }
        model_path =  args.model_path
        model = SmilesClassificationModel("bert", 
                                          model_path, 
                                          num_labels=1, 
                                          args=model_args, 
                                          use_cuda=True if not debug else False,  
                                          cuda_device=0, 
                                          ignore_mismatched_sizes=True)
        model.train_model(train_df, eval_df=test_df)
        print()    