from hyperopt import fmin, tpe, hp
import pandas as pd
import os
from rxnfp.models import SmilesClassificationModel


class SmilesClassificationModelSelf(SmilesClassificationModel):
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

def cl_mlm_finetuning(lr):
    debug = True
    trn_df = pd.read_csv(os.path.abspath('../../dataset/source_dataset/USPTO_yield/gram_trn_random_split.csv'), encoding='utf-8')
    val_df = pd.read_csv(os.path.abspath('../../dataset/source_dataset/USPTO_yield/gram_val_random_split.csv'), encoding='utf-8')

    mean = trn_df.labels.mean()
    std = trn_df.labels.std()
    trn_df['labels'] = (trn_df['labels'] - mean) / std
    val_df['labels'] = (val_df['labels'] - mean) / std

    model_args = {
    'num_train_epochs': 10, 
    'overwrite_output_dir': True,
    'learning_rate': lr,  
    'gradient_accumulation_steps': 1,
    'regression': True, 
    "num_labels":1, 
    "fp16": False,
    "evaluate_during_training": True, 
    'manual_seed': 42,
    "output_dir": f"../../outputs/uspto_gram/out_{lr}",
    'best_model_dir': f"../../outputs/uspto_gram/best_out_{lr}",
    "max_seq_length": 300, 
    "train_batch_size": 16,
    "warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.1 },
    'wandb_project': 'uspto_gram_lr_opt' if not debug else False,
    "use_multiprocessing_for_evaluation": False,
    "use_multiprocessing": False
    }

    model_path =  os.path.join('../../outputs/egret_pretrain_stage_2/bestoutput')
    model = SmilesClassificationModelSelf("bert", 
                                          model_path, 
                                          num_labels=1, 
                                          args=model_args, 
                                          use_cuda=True if not debug else False, 
                                          cuda_device=0, 
                                          ignore_mismatched_sizes=True)
    model.train_model(trn_df, eval_df=val_df)

    with open(os.path.join(model_args['best_model_dir'], 'eval_results.txt'), encoding='utf-8') as f:
        loss = f.readline().strip('\n')
    best_eval_loss = eval(loss.split('= ')[1])
    return(best_eval_loss)    

if __name__ == "__main__":

    best = fmin(
        fn=cl_mlm_finetuning,
        space=hp.uniform('lr', 0.00001, 0.0001),
        algo=tpe.suggest,
        max_evals=8)
    print(best)

