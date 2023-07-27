import argparse
import pandas as pd
import sys
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
                        default=0.0029877783623271656, 
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
    ('FullCV_01', 2768),
    ('FullCV_02', 2768),
    ('FullCV_03', 2768),
    ('FullCV_04', 2768),
    ('FullCV_05', 2768),
    ('FullCV_06', 2768),
    ('FullCV_07', 2768),
    ('FullCV_08', 2768),
    ('FullCV_09', 2768),
    ('FullCV_10', 2768),
    ]    
    DISCOVERY_SPLIT = [
    ('FullCV_01', 99), ('FullCV_01', 198), ('FullCV_01', 396), ('FullCV_01', 792), ('FullCV_01', 1187), ('FullCV_01', 1978),
    ('FullCV_02', 99), ('FullCV_02', 198), ('FullCV_02', 396), ('FullCV_02', 792), ('FullCV_02', 1187), ('FullCV_02', 1978),
    ('FullCV_03', 99), ('FullCV_03', 198), ('FullCV_03', 396), ('FullCV_03', 792), ('FullCV_03', 1187), ('FullCV_03', 1978),
    ('FullCV_04', 99), ('FullCV_04', 198), ('FullCV_04', 396), ('FullCV_04', 792), ('FullCV_04', 1187), ('FullCV_04', 1978),
    ('FullCV_05', 99), ('FullCV_05', 198), ('FullCV_05', 396), ('FullCV_05', 792), ('FullCV_05', 1187), ('FullCV_05', 1978),
    ('FullCV_06', 99), ('FullCV_06', 198), ('FullCV_06', 396), ('FullCV_06', 792), ('FullCV_06', 1187), ('FullCV_06', 1978), 
    ('FullCV_07', 99), ('FullCV_07', 198), ('FullCV_07', 396), ('FullCV_07', 792), ('FullCV_07', 1187), ('FullCV_07', 1978),
    ('FullCV_08', 99), ('FullCV_08', 198), ('FullCV_08', 396), ('FullCV_08', 792), ('FullCV_08', 1187), ('FullCV_08', 1978),
    ('FullCV_09', 99), ('FullCV_09', 198), ('FullCV_09', 396), ('FullCV_09', 792), ('FullCV_09', 1187), ('FullCV_09', 1978),
    ('FullCV_10', 99), ('FullCV_10', 198), ('FullCV_10', 396), ('FullCV_10', 792), ('FullCV_10', 1187), ('FullCV_10', 1978),
    ]

    for sheet_name, split_point in NAME_SPLIT:
    # for sheet_name, split_point in DISCOVERY_SPLIT:
        print(f'#############{sheet_name}##############')
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
        'manual_seed': 42,
        "output_dir": f"../../outputs/buchwald_hartwig/{sheet_name}_split_{split_point}",
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

