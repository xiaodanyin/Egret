from rxnfp.models import SmilesLanguageModelingModel


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
}
vocab_path = './dataset/pretrain_data/vocab.txt'
args = {'config': config, 
        'vocab_path': vocab_path, 
        'wandb_project': 'egret_pretrain_stage_1',
        'train_batch_size': 32,
        'manual_seed': 42,
        "fp16": False,
        "num_train_epochs": 50,
        'max_seq_length': 256,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        'output_dir': './outputs/egret_pretrain_stage_1/output',
        'best_model_dir':'./outputs/egret_pretrain_stage_1/bestoutput',
        'learning_rate': 0.0002
       }
model = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args, cuda_device=0)
train_file = './dataset/pretrain_data/pistachio_pretraining_train.txt'
eval_file = './dataset/pretrain_data/pistachio_pretraining_eval.txt'
model.train_model(train_file=train_file, eval_file=eval_file)
