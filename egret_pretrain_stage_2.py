from egret_model import BertForYieldPretrainModel
import pandas as pd

debug = True
config = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 256,
  "initializer_range": 0.02,
  "intermediate_size": 512,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 4,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "consensus_c": 1,   
  "different_c": 0.02,  
  "mlm_c": 1,         
}
vocab_path = './dataset/pretrain_data/vocab_stage_2.txt'
args = {'config': config, 
        'vocab_path': vocab_path, 
        'wandb_project': 'egret_pretrain_stage_2' if not debug else False,
        'train_batch_size': 36 if not debug else 2,
        'manual_seed': 42,
        "fp16": False,
        "num_train_epochs": 10,
        'max_seq_length': 256,
        'evaluate_during_training': True,
        'evaluate_during_training_steps': 2000 if not debug else 2,
        'overwrite_output_dir': True,
        'output_dir': './outputs/egret_pretrain_stage_2/output',
        'best_model_dir':'./outputs/egret_pretrain_stage_2/bestoutput',
        'learning_rate': 0.00001,
        "use_multiprocessing":True,
        'use_multiprocessing_for_evaluation':False,
        'process_count': 80,
        'warmup_ratio': 0.03
       }
model = BertForYieldPretrainModel(model_type='bert', 
                                  model_name='./outputs/egret_pretrain_stage_1/bestoutput', 
                                  args=args, 
                                  use_cuda=True if not debug else False, 
                                  cuda_device=0, 
                                  train_from_mlm=True)
train_file = './dataset/pretrain_data/aug_pistachio_pretraining_train_dataset.csv'
eval_file = './dataset/pretrain_data/aug_pistachio_pretraining_eval_dataset.csv'
trn_df = pd.read_csv(train_file, encoding='utf-8')
val_df = pd.read_csv(eval_file, encoding='utf-8')
model.yield_train_model(trn_df, eval_df=val_df, dataset_group_n=2, is_recover=False)



