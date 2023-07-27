import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rxnfp.models import SmilesClassificationModel


class SmilesClassificationModelSelf(SmilesClassificationModel):
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', 
                        default=8.474121064804021e-05, 
                        help="learning rate", 
                        type=float)
    parser.add_argument('--hidden_dropout_prob', 
                        default=0.008091441584969576, 
                        help="dropout rate", 
                        type=float)
    parser.add_argument('--model_path', 
                        default='../../outputs/egret_pretrain_stage_2/bestoutput', 
                        help="model path", 
                        type=str)    
    return parser.parse_args()

if __name__ == '__main__':

    debug = True        # HTE, USPTO数据集模型性能都已经验证过
    args = get_args()  
    test_model = True

    trn_df = pd.read_csv(os.path.abspath('../../dataset/source_dataset/USPTO_yield/milligram_trn_random_split.csv'), encoding='utf-8')
    val_df = pd.read_csv(os.path.abspath('../../dataset/source_dataset/USPTO_yield/milligram_val_random_split.csv'), encoding='utf-8')
    test_df = pd.read_csv(os.path.abspath('../../dataset/source_dataset/USPTO_yield/milligram_test_random_split.tsv'), sep='\t', index_col=0)
    test_df.columns = ['text', 'labels']
    
    mean = trn_df.labels.mean()
    std = trn_df.labels.std()
    trn_df['labels'] = (trn_df['labels'] - mean) / std
    val_df['labels'] = (val_df['labels'] - mean) / std

    model_args = {
    'num_train_epochs': 10, 
    'overwrite_output_dir': True,
    'learning_rate': args.learning_rate,  
    'gradient_accumulation_steps': 1,
    'regression': True, 
    "num_labels":1, 
    "fp16": False,
    "evaluate_during_training": True, 
    'manual_seed': 42,
    "output_dir": f"../../outputs/uspto_milligram/out",
    'best_model_dir': f"../../outputs/uspto_milligram/best_out",
    "max_seq_length": 300, 
    "train_batch_size": 16,
    "warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': args.hidden_dropout_prob },
    'wandb_project': False,
    "use_multiprocessing_for_evaluation":False,
    "use_multiprocessing":False
    }
    if not test_model:
        model_path =  args.model_path
        model = SmilesClassificationModelSelf("bert", 
                                              model_path, 
                                              num_labels=1, 
                                              args=model_args, 
                                              use_cuda=True if not debug else False, 
                                              cuda_device=1, 
                                              ignore_mismatched_sizes=True)
        model.train_model(trn_df, eval_df=val_df)
    else:
        model_path = os.path.join(model_args['best_model_dir'])
        model = SmilesClassificationModelSelf("bert", 
                                              model_path, 
                                              num_labels=1, 
                                              args=model_args, 
                                              use_cuda=True if not debug else False, 
                                              cuda_device=2)
        y_test = test_df.labels.values 
        y_preds = model.predict(test_df.text.values.tolist())[0]
        y_preds = y_preds * std + mean 
        y_preds = np.clip(y_preds, 0, 100)
        r_squared = r2_score(y_test, y_preds)
        rmse = mean_squared_error(y_test, y_preds) ** 0.5
        mae = mean_absolute_error(y_test, y_preds)       
        print('r2:{}, rmse:{}, mae:{}'.format(r_squared, rmse, mae))




