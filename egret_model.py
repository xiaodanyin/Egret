from rxnfp.models import SmilesTokenizer, SmilesLanguageModelingModel
from transformers import BertConfig, BertForSequenceClassification, AlbertConfig, AlbertForMaskedLM, ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining
from simpletransformers.custom_models.models import ElectraForLanguageModelingModel
from simpletransformers.classification.classification_model import (MODELS_WITH_EXTRA_SEP_TOKEN,
                                                                    MODELS_WITH_ADD_PREFIX_SPACE)
from simpletransformers.language_modeling.language_modeling_utils import SimpleDataset, load_hf_dataset, mask_tokens
from simpletransformers.classification.classification_utils import (InputExample, convert_examples_to_features,)
import torch
import copy
import numpy as np
import pandas as pd
from dataclasses import asdict
import random
import os
import warnings
import logging
from transformers.models.bert.modeling_bert import BertModel
from simpletransformers.config.model_args import LanguageModelingArgs
from torch.nn import functional as F
from multiprocessing import Pool
from tqdm import tqdm
from typing import List
import math
from transformers.optimization import AdamW, Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from simpletransformers.config.utils import sweep_config_to_sweep_values
from tqdm.auto import tqdm, trange
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, RandomSampler, DataLoader, TensorDataset, SequentialSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix

class RandomGroupSampler(Sampler[int]):
    data_source: Sized
    replacement: bool
    def __init__(self, data_source: Sized, group_n: int, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.group_n = group_n
        self.generator = generator
        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))
        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        idx_range = torch.arange(0, n)
        idx_range = idx_range.reshape((-1, self.group_n))
        shuffle_idx = torch.randperm(idx_range.shape[0], generator=generator)
        idx_range = idx_range[shuffle_idx]
        idx_range = idx_range.reshape((n,))
        yield from idx_range.tolist()

    def __len__(self) -> int:
        return self.num_samples

class BertContrastiveLearningModel(BertForSequenceClassification):
    def __init__(
        self,
        config,
        ) -> None:

        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

    def forward(
            self,
            input_ids,   
            attention_mask=None,
            token_type_ids=None,
            labels=None
            ):
            if labels == None:
                assert isinstance(input_ids, list)
                assert isinstance(attention_mask, list)
                assert isinstance(token_type_ids, list)
                org_memory_cls = self.bert(
                                 input_ids[0],
                                 attention_mask=attention_mask[0],
                                 token_type_ids=token_type_ids[0]
                                 )
                pos1_memory_cls = self.bert(
                                 input_ids[1],
                                 attention_mask=attention_mask[1],
                                 token_type_ids=token_type_ids[1]
                                  )                  
                pos_cos_1 = torch.cosine_similarity(org_memory_cls[1], pos1_memory_cls[1], dim=1)   
                consensus_score = (torch.ones_like(pos_cos_1)-pos_cos_1)/2 
                canonical_cos_score_matric = torch.abs(cos_similar(org_memory_cls[1], org_memory_cls[1]))
                diagonal_cos_score_matric = torch.eye(canonical_cos_score_matric.size(0)).float().to(self.device)
                different_score = canonical_cos_score_matric - diagonal_cos_score_matric
                different_score = different_score * input_ids[0].shape[0] / (input_ids[0].shape[0] - 1)
                return consensus_score, different_score
            if labels is not None:  
                assert isinstance(input_ids, list) 
                assert isinstance(labels, list) 
                org_memory_mlm = self.bert(input_ids[0])
                pos1_memory_mlm = self.bert(input_ids[1])             
                org_prediction_scores = self.cls(org_memory_mlm[0])     
                pos1_prediction_scores = self.cls(pos1_memory_mlm[0])    
                loss_fct = CrossEntropyLoss()  
                masked_lm_loss_org = loss_fct(org_prediction_scores.view(-1, self.config.vocab_size), labels[0].contiguous().view(-1)) 
                masked_lm_loss_pos1 = loss_fct(pos1_prediction_scores.view(-1, self.config.vocab_size), labels[1].contiguous().view(-1))
                masked_lm_loss = (masked_lm_loss_org + masked_lm_loss_pos1)/2
                return masked_lm_loss

class BertForYieldPretrainModel(SmilesLanguageModelingModel):
        def __init__(
            self,
            model_type,
            model_name,
            generator_name=None,
            discriminator_name=None,
            train_files=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            **kwargs,
        ):
            MODEL_CLASSES = {
            "bert": (BertConfig, BertContrastiveLearningModel, SmilesTokenizer),
            "albert": (AlbertConfig, AlbertForMaskedLM, SmilesTokenizer)
            }      
            self.args = self._load_model_args(model_name) 
            if isinstance(args, dict):
                self.args.update_from_dict(args)
            elif isinstance(args, LanguageModelingArgs):
                self.args = args            
            if "sweep_config" in kwargs:
                self.is_sweeping = True
                sweep_config = kwargs.pop("sweep_config")
                sweep_values = sweep_config_to_sweep_values(sweep_config)
                self.args.update_from_dict(sweep_values)
            else:
                self.is_sweeping = False
            if self.args.manual_seed:
                random.seed(self.args.manual_seed)
                np.random.seed(self.args.manual_seed)
                torch.manual_seed(self.args.manual_seed)
                if self.args.n_gpu > 0:
                    torch.cuda.manual_seed_all(self.args.manual_seed)
            if self.args.local_rank != -1:
                logger.info(f"local_rank: {self.args.local_rank}")
                torch.distributed.init_process_group(backend="nccl")
                cuda_device = self.args.local_rank
            if use_cuda:
                if torch.cuda.is_available():
                    if cuda_device == -1:
                        self.device = torch.device("cuda")
                    else:
                        self.device = torch.device(f"cuda:{cuda_device}")
                else:
                    raise ValueError(
                        "'use_cuda' set to True when cuda is unavailable."
                        " Make sure CUDA is available or set use_cuda=False."
                    )
            else:
                self.device = "cpu"    
            self.results = {}
            if not use_cuda:
                self.args.fp16 = False
            self.args.model_name = model_name
            self.args.model_type = model_type
            config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
            self.tokenizer_class = tokenizer_class
            new_tokenizer = False
            train_from_mlm = kwargs.get('train_from_mlm', False)
            del kwargs['train_from_mlm']
            if self.args.vocab_path and not train_from_mlm:
                self.tokenizer = tokenizer_class(self.args.vocab_path, do_lower_case=False)
            elif self.args.tokenizer_name:
                self.tokenizer = tokenizer_class.from_pretrained(self.args.tokenizer_name, cache_dir=self.args.cache_dir)
            elif self.args.model_name:
                if self.args.model_name == "electra":
                    self.tokenizer = tokenizer_class.from_pretrained(generator_name, cache_dir=self.args.cache_dir, **kwargs)
                    self.args.tokenizer_name = self.args.model_name
                else:
                    self.tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=self.args.cache_dir, **kwargs)
                    self.args.tokenizer_name = self.args.model_name
            else:
                if not train_files:
                    raise ValueError(
                        "model_name and tokenizer_name are not specified."
                        "You must specify train_files to train a Tokenizer."
                    )
                else:
                    self.train_tokenizer(train_files)
                    new_tokenizer = True 
            if self.args.config_name:
                self.config = config_class.from_pretrained(self.args.config_name, cache_dir=self.args.cache_dir)
            elif self.args.model_name and self.args.model_name != "electra" and not train_from_mlm:
                self.config = config_class.from_pretrained(model_name, cache_dir=self.args.cache_dir, **kwargs)
            else:
                self.config = config_class(**self.args.config, **kwargs)
            if self.args.vocab_size:
                self.config.vocab_size = self.args.vocab_size
            if new_tokenizer:
                self.config.vocab_size = len(self.tokenizer)
            if train_from_mlm:
                self.config.vocab_size = len(self.tokenizer)
            if self.args.model_type == "electra":
                if generator_name:
                    self.generator_config = ElectraConfig.from_pretrained(generator_name)
                elif self.args.model_name:
                    self.generator_config = ElectraConfig.from_pretrained(
                        os.path.join(self.args.model_name, "generator_config"), **kwargs,
                    )
                else:
                    self.generator_config = ElectraConfig(**self.args.generator_config, **kwargs)
                    if new_tokenizer:
                        self.generator_config.vocab_size = len(self.tokenizer)
                if discriminator_name:
                    self.discriminator_config = ElectraConfig.from_pretrained(discriminator_name)
                elif self.args.model_name:
                    self.discriminator_config = ElectraConfig.from_pretrained(
                        os.path.join(self.args.model_name, "discriminator_config"), **kwargs,
                    )
                else:
                    self.discriminator_config = ElectraConfig(**self.args.discriminator_config, **kwargs)
                    if new_tokenizer:
                        self.discriminator_config.vocab_size = len(self.tokenizer)           
            if self.args.block_size <= 0:
                self.args.block_size = min(self.args.max_seq_length, self.tokenizer.model_max_length)
            else:
                self.args.block_size = min(self.args.block_size, self.tokenizer.model_max_length, self.args.max_seq_length)
            if self.args.model_name:
                if self.args.model_type == "electra":
                    if self.args.model_name == "electra":
                        generator_model = ElectraForMaskedLM.from_pretrained(generator_name)
                        discriminator_model = ElectraForPreTraining.from_pretrained(discriminator_name)
                        self.model = ElectraForLanguageModelingModel(
                            config=self.config,
                            generator_model=generator_model,
                            discriminator_model=discriminator_model,
                            generator_config=self.generator_config,
                            discriminator_config=self.discriminator_config,
                            tie_generator_and_discriminator_embeddings=self.args.tie_generator_and_discriminator_embeddings,
                        )
                        model_to_resize = (
                            self.model.generator_model.module
                            if hasattr(self.model.generator_model, "module")
                            else self.model.generator_model
                        )
                        model_to_resize.resize_token_embeddings(len(self.tokenizer))

                        model_to_resize = (
                            self.model.discriminator_model.module
                            if hasattr(self.model.discriminator_model, "module")
                            else self.model.discriminator_model
                        )
                        model_to_resize.resize_token_embeddings(len(self.tokenizer))
                        self.model.generator_model = generator_model
                        self.model.discriminator_model = discriminator_model
                    else:
                        self.model = model_class.from_pretrained(
                            model_name,
                            config=self.config,
                            cache_dir=self.args.cache_dir,
                            generator_config=self.generator_config,
                            discriminator_config=self.discriminator_config,
                            **kwargs,
                        )
                        self.model.load_state_dict(
                            torch.load(os.path.join(self.args.model_name, "pytorch_model.bin"), map_location=self.device)
                        )
                else:
                    self.model = model_class.from_pretrained(
                        model_name, config=self.config, cache_dir=self.args.cache_dir, **kwargs,
                    )
            else:
                logger.info(" Training language model from scratch")
                if self.args.model_type == "electra":
                    generator_model = ElectraForMaskedLM(config=self.generator_config)
                    discriminator_model = ElectraForPreTraining(config=self.discriminator_config)
                    self.model = ElectraForLanguageModelingModel(
                        config=self.config,
                        generator_model=generator_model,
                        discriminator_model=discriminator_model,
                        generator_config=self.generator_config,
                        discriminator_config=self.discriminator_config,
                        tie_generator_and_discriminator_embeddings=self.args.tie_generator_and_discriminator_embeddings,
                    )
                    model_to_resize = (
                        self.model.generator_model.module
                        if hasattr(self.model.generator_model, "module")
                        else self.model.generator_model
                    )
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))

                    model_to_resize = (
                        self.model.discriminator_model.module
                        if hasattr(self.model.discriminator_model, "module")
                        else self.model.discriminator_model
                    )
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))
                else:
                    self.config.vocab_size = self.tokenizer.vocab_size
                    self.model = model_class(config=self.config)
                    model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
                    model_to_resize.resize_token_embeddings(len(self.tokenizer))
            if model_type in ["camembert", "xlmroberta"]:
                warnings.warn(
                    f"use_multiprocessing automatically disabled as {model_type}"
                    " fails when using multiprocessing for feature conversion."
                )
                self.args.use_multiprocessing = False
            if self.args.wandb_project and not wandb_available:
                warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
                self.args.wandb_project = None
            self.config.max_seq_length = args['max_seq_length']
        def _move_model_to_device(self):
                self.model.to(self.device)
        def load_and_cache_examples(
            self, 
            examples, 
            evaluate=False, 
            no_cache=True,
            multi_label=False, 
            verbose=True, 
            silent=False
            ):
                process_count = self.args.process_count
                tokenizer = self.tokenizer
                args = self.args
                if not no_cache:
                    no_cache = args.no_cache
                if not no_cache:
                    os.makedirs(self.args.cache_dir, exist_ok=True)
                mode = "dev" if evaluate else "train"
                if args.sliding_window or self.args.model_type == "layoutlm":
                    raise NotImplementedError('not implement load_and_cache_examples')
                    cached_features_file = os.path.join(
                        args.cache_dir,
                        "cached_{}_{}_{}_{}_{}".format(
                            mode,
                            args.model_type,
                            args.max_seq_length,
                            self.num_labels,
                            len(examples),
                        ),
                    )
                    if os.path.exists(cached_features_file) and (
                        (not args.reprocess_input_data and not no_cache)
                        or (mode == "dev" and args.use_cached_eval_features and not no_cache)
                    ):
                        features = torch.load(cached_features_file)
                        if verbose:
                            logger.info(
                                f" Features loaded from cache at {cached_features_file}"
                            )
                    else:
                        if verbose:
                            logger.info(" Converting to features started. Cache is not used.")
                            if args.sliding_window:
                                logger.info(" Sliding window enabled")

                        if self.args.model_type != "layoutlm":
                            if len(examples) == 3:
                                examples = [
                                    InputExample(i, text_a, text_b, label)
                                    for i, (text_a, text_b, label) in enumerate(zip(*examples))
                                ]
                            else:
                                examples = [
                                    InputExample(i, text_a, None, label)
                                    for i, (text_a, label) in enumerate(zip(*examples))
                                ]

                        # If labels_map is defined, then labels need to be replaced with ints
                        if self.args.labels_map and not self.args.regression:
                            for example in examples:
                                if multi_label:
                                    example.label = [
                                        self.args.labels_map[label] for label in example.label
                                    ]
                                else:
                                    example.label = self.args.labels_map[example.label]

                        features = convert_examples_to_features(
                            examples,
                            args.max_seq_length,
                            tokenizer,
                            # output_mode,
                            # XLNet has a CLS token at the end
                            cls_token_at_end=bool(args.model_type in ["xlnet"]),
                            cls_token=tokenizer.cls_token,
                            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                            sep_token=tokenizer.sep_token,
                            # RoBERTa uses an extra separator b/w pairs of sentences,
                            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                            sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                            # PAD on the left for XLNet
                            pad_on_left=bool(args.model_type in ["xlnet"]),
                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                            process_count=process_count,
                            multi_label=multi_label,
                            silent=args.silent or silent,
                            use_multiprocessing=args.use_multiprocessing_for_evaluation,
                            sliding_window=args.sliding_window,
                            flatten=not evaluate,
                            stride=args.stride,
                            add_prefix_space=args.model_type in MODELS_WITH_ADD_PREFIX_SPACE,
                            # avoid padding in case of single example/online inferencing to decrease execution time
                            pad_to_max_length=bool(len(examples) > 1),
                            args=args,
                        )
                        if verbose and args.sliding_window:
                            logger.info(
                                f" {len(features)} features created from {len(examples)} samples."
                            )

                        if not no_cache:
                            torch.save(features, cached_features_file)
                    if args.sliding_window and evaluate:
                        features = [
                            [feature_set] if not isinstance(feature_set, list) else feature_set
                            for feature_set in features
                        ]
                        window_counts = [len(sample) for sample in features]
                        features = [
                            feature for feature_set in features for feature in feature_set
                        ]
                    all_input_ids = torch.tensor(
                        [f.input_ids for f in features], dtype=torch.long
                    )
                    all_input_mask = torch.tensor(
                        [f.input_mask for f in features], dtype=torch.long
                    )
                    all_segment_ids = torch.tensor(
                        [f.segment_ids for f in features], dtype=torch.long
                    )
                    if self.args.model_type == "layoutlm":
                        all_bboxes = torch.tensor(
                            [f.bboxes for f in features], dtype=torch.long
                        )

                    if output_mode == "classification":
                        all_label_ids = torch.tensor(
                            [f.label_id for f in features], dtype=torch.long
                        )
                    elif output_mode == "regression":
                        all_label_ids = torch.tensor(
                            [f.label_id for f in features], dtype=torch.float
                        )

                    if self.args.model_type == "layoutlm":
                        dataset = TensorDataset(
                            all_input_ids,
                            all_input_mask,
                            all_segment_ids,
                            all_label_ids,
                            all_bboxes,
                        )
                    else:
                        dataset = TensorDataset(
                            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
                        )

                    if args.sliding_window and evaluate:
                        return dataset, window_counts
                    else:
                        return dataset

                else:
                    dataset = YieldSimpleDataset(
                        examples,
                        tokenizer,
                        self.args,
                        mode,
                        no_cache=no_cache,
                        )   
                    return dataset                    

        def encode_sliding_window(data):
            tokenizer, line, max_seq_length, special_tokens_count, stride, no_padding = data
            tokens = tokenizer.tokenize(line)
            stride = int(max_seq_length * stride)
            token_sets = []
            if len(tokens) > max_seq_length - special_tokens_count:
                token_sets = [
                    tokens[i : i + max_seq_length - special_tokens_count]
                    for i in range(0, len(tokens), stride)
                ]
            else:
                token_sets.append(tokens)
            features = []
            if not no_padding:
                sep_token = tokenizer.sep_token_id
                cls_token = tokenizer.cls_token_id
                pad_token = tokenizer.pad_token_id
                for tokens in token_sets:
                    tokens = [cls_token] + tokens + [sep_token]
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    padding_length = max_seq_length - len(input_ids)
                    input_ids = input_ids + ([pad_token] * padding_length)
                    assert len(input_ids) == max_seq_length
                    features.append(input_ids)
            else:
                for tokens in token_sets:
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    features.append(input_ids)
            return features

        def yield_train_model(
            self, train_df, dataset_group_n, output_dir=None, show_running_loss=True, args=None, eval_df=None, verbose=True, **kwargs,
        ):
            if args:
                self.args.update_from_dict(args)
            if self.args.silent:
                show_running_loss = False
            if self.args.evaluate_during_training and eval_df is None:
                    raise ValueError(
                        "evaluate_during_training is enabled but eval_file is not specified."
                        " Pass eval_file to model.train_model() if using evaluate_during_training."
                    )
            if not output_dir:
                    output_dir = self.args.output_dir
            if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
                    raise ValueError(
                        "Output directory ({}) already exists and is not empty."
                        " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
                    )
            self._move_model_to_device()
            train_examples = (
                        train_df["org_smi"].astype(str).tolist(), 
                        train_df["positive_smi_1"].astype(str).tolist(),    
                    )
            train_dataset = self.load_and_cache_examples(train_examples, verbose=verbose)
            os.makedirs(output_dir, exist_ok=True)
            global_step, training_details = self.yield_train(
                train_dataset,
                # train_dataloader,
                output_dir, 
                dataset_group_n=dataset_group_n,
                show_running_loss=show_running_loss,
                eval_df=eval_df,
                verbose=verbose,
                **kwargs,
            ) 
            self.save_model(output_dir, model=self.model)
            if self.args.model_type == "electra":
                self.save_discriminator()
                self.save_generator()
            if verbose:
                logger.info(
                    " Training of {} model complete. Saved to {}.".format(
                        self.args.model_type, output_dir
                    )
                )
            return global_step, training_details
        
        def _create_training_progress_scores(self, **kwargs):
            extra_metrics = {key: [] for key in kwargs}
            training_progress_scores = {
                "global_step": [],
                "perplexity": [],
                "eval_loss": [],
                "train_loss": [],
                "eval_consensus_loss":  [],
                "eval_difference_loss": [],
                "eval_loss_mlm": [],
                **extra_metrics,
            }
            return training_progress_scores
        
        def yield_train(
            self, train_dataset, output_dir, dataset_group_n, show_running_loss=True, eval_df=None, verbose=True, **kwargs,
        ):
            model = self.model
            args = self.args
            tokenizer = self.tokenizer

            if self.is_world_master():
                tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
            train_sampler = RandomGroupSampler(train_dataset, group_n=dataset_group_n) if args.local_rank == -1 else DistributedSampler(train_dataset) 
            if self.args.use_hf_datasets:
                train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler)
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler)
            if args.max_steps > 0:
                t_total = args.max_steps
                args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
            else:
                t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = []
            custom_parameter_names = set()
            for group in self.args.custom_parameter_groups:
                params = group.pop("params")
                custom_parameter_names.update(params)
                param_group = {**group}
                param_group["params"] = [p for n, p in model.named_parameters() if n in params]
                optimizer_grouped_parameters.append(param_group)
            for group in self.args.custom_layer_parameters:
                layer_number = group.pop("layer")
                layer = f"layer.{layer_number}."
                group_d = {**group}
                group_nd = {**group}
                group_nd["weight_decay"] = 0.0
                params_d = []
                params_nd = []
                for n, p in model.named_parameters():
                    if n not in custom_parameter_names and layer in n:
                        if any(nd in n for nd in no_decay):
                            params_nd.append(p)
                        else:
                            params_d.append(p)
                        custom_parameter_names.add(n)
                group_d["params"] = params_d
                group_nd["params"] = params_nd
                optimizer_grouped_parameters.append(group_d)
                optimizer_grouped_parameters.append(group_nd)
            if not self.args.train_custom_parameters_only:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                )
            is_recover = kwargs.get('is_recover', True)
            del kwargs['is_recover']
            warmup_steps = math.ceil(t_total * args.warmup_ratio)
            if is_recover:
                args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps
            else:
                args.warmup_steps = warmup_steps
            if args.optimizer == "AdamW":
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            elif args.optimizer == "Adafactor":
                optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    eps=args.adafactor_eps,
                    clip_threshold=args.adafactor_clip_threshold,
                    decay_rate=args.adafactor_decay_rate,
                    beta1=args.adafactor_beta1,
                    weight_decay=args.weight_decay,
                    scale_parameter=args.adafactor_scale_parameter,
                    relative_step=args.adafactor_relative_step,
                    warmup_init=args.adafactor_warmup_init,
                )
                print("Using Adafactor for T5")
            else:
                raise ValueError(
                    "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                        args.optimizer
                    )
                )
            if args.scheduler == "constant_schedule":
                scheduler = get_constant_schedule(optimizer)
            elif args.scheduler == "constant_schedule_with_warmup":
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
            elif args.scheduler == "linear_schedule_with_warmup":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
                )
            elif args.scheduler == "cosine_schedule_with_warmup":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=t_total,
                    num_cycles=args.cosine_schedule_num_cycles,
                )
            elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
                scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=t_total,
                    num_cycles=args.cosine_schedule_num_cycles,
                )
            elif args.scheduler == "polynomial_decay_schedule_with_warmup":
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=t_total,
                    lr_end=args.polynomial_decay_schedule_lr_end,
                    power=args.polynomial_decay_schedule_power,
                )
            else:
                raise ValueError("{} is not a valid scheduler.".format(args.scheduler))
            if (
                args.model_name
                and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
                and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
            ): 
                if is_recover:
                    # Load in optimizer and scheduler states
                    optimizer.load_state_dict(torch.load(os.path.join(args.model_name, "optimizer.pt"), map_location=self.device))
                    scheduler.load_state_dict(torch.load(os.path.join(args.model_name, "scheduler.pt"), map_location=self.device))
                else:
                    print('Optimizer and scheduler was not loaded, start traing step from default initialization.')
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            # Distributed training
            if args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
                )
            logger.info(" Training started")
            global_step = 0
            training_progress_scores = None
            tr_loss, logging_loss, tr_consensus_loss,  tr_difference_loss, tr_loss_mlm = 0.0, 0.0, 0.0, 0.0, 0.0
            model.zero_grad()
            train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
            epoch_number = 0
            best_eval_metric = None
            early_stopping_counter = 0
            steps_trained_in_current_epoch = 0
            epochs_trained = 0
            if args.model_name and os.path.exists(args.model_name):
                try:
                    # set global_step to gobal_step of last saved checkpoint from model path
                    checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                    if len(checkpoint_suffix) > 2:
                        checkpoint_suffix = checkpoint_suffix[1]
                    else:
                        checkpoint_suffix = checkpoint_suffix[-1]
                    global_step = int(checkpoint_suffix)
                    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                    steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                    )
                    logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                    logger.info("   Continuing training from epoch %d", epochs_trained)
                    logger.info("   Continuing training from global step %d", global_step)
                    logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
                except ValueError:
                    logger.info("   Starting fine-tuning.")
            if args.evaluate_during_training:
                training_progress_scores = self._create_training_progress_scores(**kwargs)
            if args.wandb_project:
                wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
                wandb.run._label(repo="simpletransformers")
                wandb.watch(self.model)
            if args.fp16:
                from torch.cuda import amp
                scaler = amp.GradScaler()
            for current_epoch in train_iterator:
                model.train()
                if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(current_epoch)
                if epochs_trained > 0:
                    epochs_trained -= 1
                    continue
                train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
                batch_iterator = tqdm(
                    train_dataloader,
                    desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                    disable=args.silent,
                    mininterval=0,
                )
                for step, batch_cls in enumerate(batch_iterator):
                    batch_mlm = copy.deepcopy(batch_cls)
                    batch_mlm = batch_mlm[0]['input_ids']                
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue
                    inputs_cls = self._get_inputs_dict(batch_cls)
                    cl_input_id = [
                        inputs_cls['input_ids'][:, 0: args.max_seq_length],
                        inputs_cls['input_ids'][:, args.max_seq_length : 2* args.max_seq_length],
                    ]
                    cl_token_type_ids = [
                        inputs_cls['token_type_ids'][:, 0: args.max_seq_length],
                        inputs_cls['token_type_ids'][:, args.max_seq_length : 2* args.max_seq_length],                     
                    ]
                    cl_attention_mask = [
                        inputs_cls['attention_mask'][:, 0: args.max_seq_length],
                        inputs_cls['attention_mask'][:, args.max_seq_length : 2* args.max_seq_length],                      
                    ]
                    mlm_batch_org = batch_mlm[:, 0: args.max_seq_length]
                    mlm_batch_pos1 = batch_mlm[:, args.max_seq_length : 2* args.max_seq_length]
                    inputs_mlm_org, labels_org = mask_tokens(mlm_batch_org, tokenizer, args) if args.mlm else (mlm_batch_org, mlm_batch_org)
                    inputs_mlm_pos1, labels_pos1 = mask_tokens(mlm_batch_pos1, tokenizer, args) if args.mlm else (mlm_batch_pos1, mlm_batch_pos1)
                    inputs_mlm = [inputs_mlm_org, inputs_mlm_pos1]
                    inputs_mlm = [x.to(self.device) for x in inputs_mlm]
                    labels = [labels_org, labels_pos1]
                    labels = [x.to(self.device) for x in labels]
                    if self.args.fp16:
                        with amp.autocast():
                            consensus_score, different_score = model(cl_input_id, cl_token_type_ids, cl_attention_mask)
                            loss_mlm = model(inputs_mlm, labels=labels) if args.mlm else model(inputs_mlm, labels=labels)
                    else:
                        consensus_score, different_score = model(cl_input_id, cl_token_type_ids, cl_attention_mask)
                        loss_mlm = model(inputs_mlm, labels=labels) if args.mlm else model(inputs_mlm, labels=labels)
                    if not hasattr(self.config, 'consensus_c'):
                        consensus_loss = consensus_score.mean()
                        difference_loss = different_score.mean()
                        loss =  consensus_loss + difference_loss +  loss_mlm
                    else:
                        consensus_loss = self.config.consensus_c * consensus_score.mean()
                        difference_loss = self.config.different_c * different_score.mean()
                        loss_mlm = self.config.mlm_c * loss_mlm
                        loss = consensus_loss + difference_loss + loss_mlm
                    if args.n_gpu > 1:
                        loss = loss.mean() 
                    current_loss = loss.item()
                    if show_running_loss:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f},  Consensus Loss: {consensus_loss.item():9.4f}, Difference Loss: {difference_loss.item():9.4f}, MLM Loss: {loss_mlm.item():9.4f}"
                        )
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps                    
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    tr_loss += loss.item()
                    tr_consensus_loss += consensus_loss.item()
                    tr_difference_loss += difference_loss.item()
                    tr_loss_mlm += loss_mlm.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            scaler.unscale_(optimizer)
                        if args.optimizer == "AdamW":
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        if args.fp16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()  
                        model.zero_grad()
                        global_step += 1
                        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            if self.is_world_master():
                                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                            logging_loss = tr_loss
                            if args.wandb_project or self.is_sweeping:
                                wandb.log(
                                    {
                                        "Training loss": current_loss,
                                        "lr": scheduler.get_last_lr()[0],
                                        "global_step": global_step,
                                    }
                                )
                        if args.save_steps > 0 and global_step % args.save_steps == 0:
                            output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                            self.save_model(output_dir_current, optimizer, scheduler, model=model)
                        if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                        ):
                            results = self.yield_eval_model(
                                # eval_file,
                                eval_df,
                                verbose=verbose and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                **kwargs,
                            )
                            if self.is_world_master():
                                for key, value in results.items():
                                    try:
                                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                                    except (NotImplementedError, AssertionError):
                                        pass
                            output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                            if args.save_eval_checkpoints:
                                self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)
                            training_progress_scores["global_step"].append(global_step)
                            training_progress_scores["train_loss"].append(current_loss)
                            for key in results:
                                training_progress_scores[key].append(results[key])
                            report = pd.DataFrame(training_progress_scores)
                            report.to_csv(
                                os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                            )
                            if args.wandb_project or self.is_sweeping:
                                wandb.log(self._get_last_metrics(training_progress_scores))
                            if not best_eval_metric:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                            if best_eval_metric and args.early_stopping_metric_minimize:
                                if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                    best_eval_metric = results[args.early_stopping_metric]
                                    self.save_model(
                                            args.best_model_dir, optimizer, scheduler, model=model, results=results
                                        )
                                    early_stopping_counter = 0
                                else:
                                    if args.use_early_stopping:
                                        if early_stopping_counter < args.early_stopping_patience:
                                            early_stopping_counter += 1
                                            if verbose:
                                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                                    logger.info(f" Current step: {early_stopping_counter}")
                                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                            else:
                                                if verbose:
                                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached.")
                                                    logger.info(" Training terminated.")
                                                    train_iterator.close()
                                                return (
                                                    global_step,
                                                    tr_loss / global_step
                                                    if not self.args.evaluate_during_training
                                                    else training_progress_scores,
                                                )
                            else:
                                if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                    best_eval_metric = results[args.early_stopping_metric]
                                    self.save_model(
                                            args.best_model_dir, optimizer, scheduler, model=model, results=results
                                        )
                                    early_stopping_counter = 0
                                else:
                                    if args.use_early_stopping:
                                        if early_stopping_counter < args.early_stopping_patience:
                                            early_stopping_counter += 1
                                            if verbose:
                                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                                    logger.info(f" Current step: {early_stopping_counter}")
                                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                            else:
                                                if verbose:
                                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached.")
                                                    logger.info(" Training terminated.")
                                                    train_iterator.close()
                                                return (
                                                    global_step,
                                                    tr_loss / global_step
                                                    if not self.args.evaluate_during_training
                                                    else training_progress_scores,
                                                )
                                model.train()
                    if args.max_steps > 0 and global_step > args.max_steps:
                            return (
                                global_step,
                                tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
                            )
                epoch_number += 1
                output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))                    
                if args.save_model_every_epoch or args.evaluate_during_training:
                    os.makedirs(output_dir_current, exist_ok=True)
                if args.save_model_every_epoch:
                    self.save_model(output_dir_current, optimizer, scheduler, model=model)        
                if args.evaluate_during_training and args.evaluate_each_epoch:
                    results = self.yield_eval_model(
                        # eval_file,
                        eval_df,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        **kwargs,
                    )
                    self.save_model(output_dir_current, optimizer, scheduler, results=results)
                    training_progress_scores["global_step"].append(global_step)
                    training_progress_scores["train_loss"].append(current_loss)
                    for key in results:
                        training_progress_scores[key].append(results[key])
                    report = pd.DataFrame(training_progress_scores)
                    report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)
                    if args.wandb_project or self.is_sweeping:
                        wandb.log(self._get_last_metrics(training_progress_scores))
                    if not best_eval_metric:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                    if best_eval_metric and args.early_stopping_metric_minimize:
                        if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping and args.early_stopping_consider_epochs:
                                if early_stopping_counter < args.early_stopping_patience:
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(f" No improvement in {args.early_stopping_metric}")
                                        logger.info(f" Current step: {early_stopping_counter}")
                                        logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                else:
                                    if verbose:
                                        logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                    else:
                        if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                            early_stopping_counter = 0
                        else:
                            if args.use_early_stopping and args.early_stopping_consider_epochs:
                                if early_stopping_counter < args.early_stopping_patience:
                                    early_stopping_counter += 1
                                    if verbose:
                                        logger.info(f" No improvement in {args.early_stopping_metric}")
                                        logger.info(f" Current step: {early_stopping_counter}")
                                        logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                else:
                                    if verbose:
                                        logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                        logger.info(" Training terminated.")
                                        train_iterator.close()
                                    return (
                                        global_step,
                                        tr_loss / global_step
                                        if not self.args.evaluate_during_training
                                        else training_progress_scores,
                                    )
                if args.max_steps > 0 and global_step > args.max_steps:
                    return (
                        global_step,
                        tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
                    )    
            return (
                global_step,
                tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
            )
        def yield_eval_model(self, eval_df, output_dir=None, verbose=True, silent=False, **kwargs):
            if not output_dir:
                output_dir = self.args.output_dir
            self._move_model_to_device()
            eval_examples = (
                        eval_df["org_smi"].astype(str).tolist(), 
                        eval_df["positive_smi_1"].astype(str).tolist(),
                    )
            eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True, verbose=verbose, silent=silent)
            os.makedirs(output_dir, exist_ok=True)
            result = self.yield_evaluate(eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs)
            self.results.update(result)
            if verbose:
                logger.info(self.results)
            return result
    
        def yield_evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, **kwargs):
            model = self.model
            args = self.args
            eval_output_dir = output_dir
            tokenizer = self.tokenizer   
            results = {}
            eval_sampler = SequentialSampler(eval_dataset)  
            if self.args.use_hf_datasets:
                eval_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size, sampler=eval_sampler)
            else:
                eval_dataloader = DataLoader(eval_dataset, batch_size=args.train_batch_size, sampler=eval_sampler)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)    
            eval_loss, eval_consensus_loss, eval_difference_loss, eval_loss_mlm = 0.0, 0.0, 0.0, 0.0
            nb_eval_steps = 0
            model.eval() 
            for batch_cls in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
                if self.args.use_hf_datasets:
                    batch = batch["input_ids"] 
                batch_mlm = copy.deepcopy(batch_cls)
                batch_mlm = batch_mlm[0]['input_ids']
                inputs_cls = self._get_inputs_dict(batch_cls)
                cl_input_id = [
                    inputs_cls['input_ids'][:, 0: args.max_seq_length],
                    inputs_cls['input_ids'][:, args.max_seq_length : 2* args.max_seq_length],
                ]
                cl_token_type_ids = [
                    inputs_cls['token_type_ids'][:, 0: args.max_seq_length],
                    inputs_cls['token_type_ids'][:, args.max_seq_length : 2* args.max_seq_length],                   
                ]
                cl_attention_mask = [
                    inputs_cls['attention_mask'][:, 0: args.max_seq_length],
                    inputs_cls['attention_mask'][:, args.max_seq_length : 2* args.max_seq_length],                    
                ]                
                mlm_batch_org = batch_mlm[:, 0: args.max_seq_length]
                mlm_batch_pos1 = batch_mlm[:, args.max_seq_length : 2* args.max_seq_length]
                inputs_mlm_org, labels_org = mask_tokens(mlm_batch_org, tokenizer, args) if args.mlm else (mlm_batch_org, mlm_batch_org)
                inputs_mlm_pos1, labels_pos1 = mask_tokens(mlm_batch_pos1, tokenizer, args) if args.mlm else (mlm_batch_pos1, mlm_batch_pos1)
                inputs_mlm = [inputs_mlm_org, inputs_mlm_pos1]
                inputs_mlm = [x.to(self.device) for x in inputs_mlm]
                labels = [labels_org, labels_pos1]
                labels = [x.to(self.device) for x in labels]               
                with torch.no_grad(): 
                    consensus_score, different_score = model(cl_input_id, cl_token_type_ids, cl_attention_mask)
                    loss_mlm = model(inputs_mlm, labels=labels) if args.mlm else model(inputs_mlm, labels=labels)
                    if not hasattr(self.config, 'consensus_c'):
                        consensus_loss = consensus_score.mean()
                        difference_loss = different_score.mean()
                        loss =  consensus_loss + difference_loss +  loss_mlm
                    else:
                        consensus_loss = self.config.consensus_c * consensus_score.mean()
                        difference_loss = self.config.different_c * different_score.mean()
                        loss_mlm = self.config.mlm_c * loss_mlm
                        loss = consensus_loss + difference_loss + loss_mlm
                    if self.args.n_gpu > 1:
                        loss = loss.mean()   
                    eval_loss += loss.item()
                    eval_consensus_loss += consensus_loss.item()
                    eval_difference_loss += difference_loss.item()
                    eval_loss_mlm += loss_mlm.item()
                nb_eval_steps += 1            
            eval_loss = eval_loss / nb_eval_steps
            eval_consensus_loss = eval_consensus_loss / nb_eval_steps
            eval_difference_loss = eval_difference_loss / nb_eval_steps
            eval_loss_mlm = eval_loss_mlm / nb_eval_steps
            perplexity = torch.exp(torch.tensor(eval_loss))
            results["eval_loss"] = eval_loss
            results["eval_consensus_loss"] = eval_consensus_loss
            results["eval_difference_loss"] = eval_difference_loss
            results["eval_loss_mlm"] = eval_loss_mlm
            results["perplexity"] = perplexity   
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt") 
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
            return results
                        
        def _get_inputs_dict(self, batch, no_hf=False):
            if self.args.use_hf_datasets and not no_hf:
                return {key: value.to(self.device) for key, value in batch.items()}
            if isinstance(batch[0], dict):
                inputs = {key: value.squeeze(1).to(self.device) for key, value in batch[0].items()}
                # inputs["labels"] = batch[1].to(self.device)
            else:
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2]
                        if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"]
                        else None
                    )
            if self.args.model_type == "layoutlm":
                inputs["bbox"] = batch[4]
            return inputs

def preprocess_data_multiprocessing(data):
    text_a, text_b, tokenizer, max_seq_length = data
    examples = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return examples

def preprocess_data(text_a, text_b, tokenizer, max_seq_length):
    return tokenizer(
        text=text_a,
        text_pair=text_b,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )

def build_classification_dataset(
    data, tokenizer, args, mode, multi_label,
):
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            args.model_type,
            args.max_seq_length,
            # len(args.labels_list),
            len(data),
        ),
    )
    if os.path.exists(cached_features_file) and (
        (not args.reprocess_input_data and not args.no_cache)
        or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
    ):
        data = torch.load(cached_features_file)
        logger.info(f" Features loaded from cache at {cached_features_file}")
        examples, labels = data
    else:
        logger.info(" Converting to features started. Cache is not used.")
        text_org, text_aug1  = data
        text_b = None
        if (mode == "train" and args.use_multiprocessing) or (
            mode == "dev" and args.use_multiprocessing_for_evaluation
        ):
            if args.multiprocessing_chunksize == -1:
                chunksize = max(len(data) // (args.process_count * 2), 500)
            else:
                chunksize = args.multiprocessing_chunksize
            if text_b is not None:
                data = [
                    (
                        text_a[i : i + chunksize],
                        text_b[i : i + chunksize],
                        tokenizer,
                        args.max_seq_length,
                    )
                    for i in range(0, len(text_a), chunksize)
                ]
            else:
                data_org = [
                    (text_org[i : i + chunksize], None, tokenizer, args.max_seq_length)
                    for i in range(0, len(text_org), chunksize)
                ]   
                data_aug1 = [
                    (text_aug1[i : i + chunksize], None, tokenizer, args.max_seq_length)
                    for i in range(0, len(text_aug1), chunksize)
                ]
            with Pool(args.process_count) as p:
                examples_org = list(
                    tqdm(
                        p.imap(preprocess_data_multiprocessing, data_org),
                        total=len(data_org),
                        disable=args.silent,
                    )
                )
            examples_org = {
                key: torch.cat([example[key] for example in examples_org])
                for key in examples_org[0]
            }
            with Pool(args.process_count) as p:
                examples_aug1 = list(
                    tqdm(
                        p.imap(preprocess_data_multiprocessing, data_aug1),
                        total=len(data_aug1),
                        disable=args.silent,
                    )
                )
            examples_aug1 = {
                key: torch.cat([example[key] for example in examples_aug1])
                for key in examples_aug1[0]
            }                                
        else:
            examples_org = preprocess_data(text_org, text_b, tokenizer, args.max_seq_length)
            examples_aug1 = preprocess_data(text_aug1, text_b, tokenizer, args.max_seq_length)
        input_ids = torch.cat((examples_org['input_ids'], examples_aug1['input_ids']), 1)
        token_type_ids = torch.cat((examples_org['token_type_ids'], examples_aug1['token_type_ids']), 1)
        attention_mask = torch.cat((examples_org['attention_mask'], examples_aug1['attention_mask']), 1)
        examples = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
    return examples

class YieldSimpleDataset(Dataset):
    def __init__(self, data, tokenizer, args, mode, no_cache):
        self.examples = build_classification_dataset(data, tokenizer, args, mode, no_cache)
    def __len__(self):
        return len(self.examples["input_ids"])
    def __getitem__(self, index):
        return ({key: self.examples[key][index] for key in self.examples},)

                             
                
