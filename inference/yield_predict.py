import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from inference.models import SmilesClassificationModel
import numpy as np
import torch
from argparse import ArgumentParser

class SmilesClassificationModelSelf(SmilesClassificationModel):
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

class YieldPredictorAPI:
    def __init__(self, model_state_path, cuda_device=-1) -> None:
        self.model_state_path = model_state_path
        use_cuda = True if torch.cuda.is_available() else False
        if cuda_device == -1:
            use_cuda = False
        self.model_class = SmilesClassificationModelSelf(
            "bert", 
            self.model_state_path, 
            num_labels=1, 
            use_cuda=use_cuda, 
            cuda_device=cuda_device)
        self.yield_mean = 79.29119663076209   
        self.yield_std = 18.858441890553195   
    
    def _to_yield_class(self, y_preds):
        y_preds_cls = -np.ones_like(y_preds)
        y_preds_cls[np.argwhere((y_preds>=80)&(y_preds<100))] = 0
        y_preds_cls[np.argwhere((y_preds>=50)&(y_preds<80))] = 1
        y_preds_cls[np.argwhere((y_preds>=30)&(y_preds<50))] = 2
        y_preds_cls[np.argwhere((y_preds>=0)&(y_preds<30))] = 3
        return y_preds_cls.astype(int)

        
    def predict(self, rxn_smiles_list, to_yield_class=False):
        y_preds = self.model_class.predict(rxn_smiles_list)[0]
        y_preds = y_preds * self.yield_std + self.yield_mean 
        y_preds = np.clip(y_preds, 0, 100)
        if to_yield_class:
            y_preds_cls = self._to_yield_class(y_preds)
            result_dic = {
                'rxn_smiles': [],
                'predicted_yield_class': []
            }
            result_dic['rxn_smiles'] = rxn_smiles_list
            result_dic['predicted_yield_class'] = y_preds_cls
            result_df = pd.DataFrame.from_dict(result_dic)
            result_df.to_csv(parser_args.output_path, index=False)
        else:
            result_dic = {
                'rxn_smiles': [],
                'predicted_yield': []
            }
            result_dic['rxn_smiles'] = rxn_smiles_list
            result_dic['predicted_yield'] = y_preds
            result_df = pd.DataFrame.from_dict(result_dic)
            result_df.to_csv(parser_args.output_path, index=False)            

def main(parser_args):
    with open(parser_args.input_path, 'r', encoding='utf-8') as f:
        reaction = [x.strip() for x in f.readlines()]
    this_path = os.path.abspath(os.path.dirname(__file__))
    model_state_path = os.path.join(this_path, 'yield_prediction_model')
    yield_predictor = YieldPredictorAPI(model_state_path=model_state_path)
    yield_predictor.predict(reaction, to_yield_class=False)


        

if __name__ == "__main__":

    parser = ArgumentParser('Test Arguements')
    parser.add_argument('--input_path',
                        default='test_files/input_demo.txt',
                        help='Path to input file (txt)',
                        type=str)
    parser.add_argument('--output_path',
                        default='test_files/predicted_yields.csv',
                        help='Path to output file (csv)',
                        type=str)
    parser_args = parser.parse_args()

    main(parser_args)

