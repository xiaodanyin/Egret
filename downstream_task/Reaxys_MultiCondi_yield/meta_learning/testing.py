import pandas as pd
import os
import numpy as np
from rxnfp.models import SmilesClassificationModel

class SmilesClassificationModelSelf(SmilesClassificationModel):
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

if __name__ == '__main__':

    debug = True
    test_model = True

    # trn_df = pd.read_csv('../../../data/all_yield_data/without_time_grouping/labled_trn_dataset_.csv', encoding = 'utf-8')
    # val_df = pd.read_csv('../../../data/all_yield_data/without_time_grouping/labled_val_dataset_.csv', encoding = 'utf-8')
    test_df = pd.read_csv('../../../data/all_yield_data/without_time_grouping/test_dataset_category_0.csv', encoding = 'utf-8')

    # trn_df = trn_df[['fin_can_rxn_smi', 'yield_label']]
    # val_df = val_df[['fin_can_rxn_smi', 'yield_label']]
    test_df = test_df[['fin_can_rxn_smi', 'yield_label']]

    # trn_df.columns = ['text', 'labels']
    # val_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']

    model_args = {
    'num_train_epochs': 20, 
    'overwrite_output_dir': True,
    'learning_rate': 1.0e-04,  
    'gradient_accumulation_steps': 1,
    'regression': True, 
    "num_labels":1, 
    "fp16": False,
    "evaluate_during_training": True, 
    'manual_seed': 42,
    "output_dir": "../../outputs/reaxys_yield/regression/out",
    'best_model_dir': "../../outputs/reaxys_yield/regression/bestout",
    "max_seq_length": 300, 
    "train_batch_size": 16,
    "warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.1 },
    'wandb_project': False,
    "use_multiprocessing_for_evaluation":False,
    "use_multiprocessing":False
    }
    if not test_model:

        model_path =  os.path.join('../../../outputs/egret_pretrain_stage_2/bestoutput')
        model = SmilesClassificationModelSelf("bert", model_path, num_labels=4, 
                            args=model_args, use_cuda=True if not debug else False, cuda_device=2, ignore_mismatched_sizes=True)

        model.train_model(trn_df, eval_df=val_df)
    else:
    
        model_path = os.path.join('../../../outputs/meta_learning_output/category_0/best_model') 
        model = SmilesClassificationModelSelf("bert", model_path, num_labels=4, 
                            args=model_args, use_cuda=True, cuda_device=2)

        y_test = test_df.labels.values    
        y_preds = model.predict(test_df.text.values.tolist())[0]
        accuracy = (y_test == np.array(y_preds)).sum()/y_test.shape[0]
        print(accuracy)






    




