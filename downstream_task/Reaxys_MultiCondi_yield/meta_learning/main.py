import json
from random import shuffle
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import json
from torch.utils.data import TensorDataset
import torch
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from copy import deepcopy
import gc
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import shutil
from rxnfp.models import SmilesClassificationModel



def copy_model_state_file(source_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)
                print(src_file)
    
class SmilesClassificationModelSelf(SmilesClassificationModel):
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

class MetaTask(Dataset):
    
    def __init__(self, examples, num_task, k_support, k_query, tokenizer):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples        
        random.shuffle(self.examples)
        
        self.num_task = num_task    
        self.k_support = k_support  
        self.k_query = k_query      
        self.tokenizer = tokenizer
        self.max_seq_length = 300   
        self.create_batch(self.num_task)
    
    def create_batch(self, num_task):       
        self.supports = []  
        self.queries = []  
        
        for b in range(num_task):  
            # 1.select domain randomly
            domain = random.choice(self.examples)['domain']     
            domainExamples = [e for e in self.examples if e['domain'] == domain]
            
            # 1.select k_support + k_query examples from domain randomly
            selected_examples = random.sample(domainExamples,self.k_support + self.k_query)
            random.shuffle(selected_examples)
            exam_train = selected_examples[:self.k_support]     
            exam_test  = selected_examples[self.k_support:]
            
            self.supports.append(exam_train)
            self.queries.append(exam_test)

    def create_feature_set(self, examples):
        all_input_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)       
        all_attention_mask = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)        
        all_segment_ids    = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)        
        all_label_ids      = torch.empty(len(examples), dtype = torch.long)                            

        for id_, example in enumerate(examples):
            input_ids = tokenizer.encode(example['text'])
            attention_mask = [1] * len(input_ids)
            segment_ids    = [0] * len(input_ids)

            if len(input_ids) > self.max_seq_length:         
                input_ids = input_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]
                segment_ids = segment_ids[:self.max_seq_length]
            
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            label_id = example['label']

            all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(attention_mask).to(torch.long)
            all_segment_ids[id_] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[id_] = torch.Tensor([label_id]).to(torch.long)

        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)  
        return tensor_set
    
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index])
        query_set   = self.create_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task
    
class MetaTestingTask(Dataset):
    
    def __init__(self, support_examples, query_examples, num_task, k_support, k_query, tokenizer):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.support_examples = support_examples        
        random.shuffle(self.support_examples)

        self.query_examples = query_examples       
        random.shuffle(self.query_examples)        
        
        self.num_task = num_task      
        self.k_support = k_support  
        self.k_query = k_query      
        self.tokenizer = tokenizer
        self.max_seq_length = 300  
        self.create_batch(self.num_task)
    
    def create_batch(self, num_task):       
        self.supports = []  
        self.queries = []  

        exam_train = self.support_examples     
        exam_test  = self.query_examples

        self.supports.append(exam_train)
        self.queries.append(exam_test)        

    def create_feature_set(self, examples):
        all_input_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_attention_mask = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_segment_ids    = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_label_ids      = torch.empty(len(examples), dtype = torch.long)

        for id_, example in enumerate(examples):
            input_ids = tokenizer.encode(example['text'])
            attention_mask = [1] * len(input_ids)
            segment_ids    = [0] * len(input_ids)

            if len(input_ids) > self.max_seq_length:         
                input_ids = input_ids[:self.max_seq_length]
                attention_mask = attention_mask[:self.max_seq_length]
                segment_ids = segment_ids[:self.max_seq_length]

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            label_id = example['label']
            all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(attention_mask).to(torch.long)
            all_segment_ids[id_] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[id_] = torch.Tensor([label_id]).to(torch.long)

        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)  
        return tensor_set
    
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index])
        query_set   = self.create_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task


def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0, len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size, len(taskset)))]

class TrainingArgs:
    def __init__(self):
        self.num_labels = 4     
        self.meta_epoch = 10
        self.k_spt = 80
        self.k_qry = 20
        self.outer_batch_size = 2
        self.inner_batch_size = 12
        self.outer_update_lr = 0.00001
        self.inner_update_lr = 0.00001
        self.inner_update_step = 10
        self.inner_update_step_eval = 20      
        self.bert_model = '../../../outputs/egret_pretrain_stage_2/bestoutput'
        self.num_task_train = 100
        self.num_task_test = 5

# Create Meta Learner
class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
                                                                
        self.num_labels = args.num_labels                      
        self.outer_batch_size = args.outer_batch_size           
        self.inner_batch_size = args.inner_batch_size          
        self.outer_update_lr  = args.outer_update_lr           
        self.inner_update_lr  = args.inner_update_lr            
        self.inner_update_step = args.inner_update_step         
        self.inner_update_step_eval = args.inner_update_step_eval
        self.bert_model = args.bert_model
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.model = SmilesClassificationModelSelf('bert', self.bert_model, num_labels = self.num_labels, cuda_device=2, ignore_mismatched_sizes=True)
        self.outer_optimizer = Adam(self.model.model.parameters(), lr=self.outer_update_lr)
        self.model.model.train()

    def forward(self, batch_tasks, training = True):
        """
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
        
        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval

        if training:
            for task_id, task in enumerate(batch_tasks):  
                support = task[0]
                query   = task[1]
                
                fast_model = deepcopy(self.model.model)
                fast_model = fast_model.to(self.device)
                support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                                batch_size=self.inner_batch_size)
                
                inner_optimizer = Adam(fast_model.parameters(), lr=self.inner_update_lr)
                fast_model.train()
                
                print('----Task', task_id, '----')
                for i in range(0, num_inner_update_step):   
                    all_loss = []
                    for inner_step, batch in enumerate(support_dataloader):     
                        
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, segment_ids, label_id = batch
                        outputs = fast_model(input_ids, attention_mask, segment_ids, labels = label_id)
                        
                        loss = outputs[0]              
                        loss.backward()
                        inner_optimizer.step()
                        inner_optimizer.zero_grad()
                        
                        all_loss.append(loss.item()) 
                    
                    if i % 4 == 0:
                        print("Inner Loss: ", np.mean(all_loss))
                
                fast_model = fast_model.to(torch.device('cpu'))
                
                if training:
                    meta_weights = list(self.model.model.parameters())
                    fast_weights = list(fast_model.parameters())

                    gradients = []
                    for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
                        gradient = meta_params - fast_params
                        if task_id == 0:
                            sum_gradients.append(gradient)
                        else:
                            sum_gradients[i] += gradient

                fast_model = fast_model.to(self.device)
                fast_model.eval()
                with torch.no_grad():
                    query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
                    query_batch = iter(query_dataloader).next()
                    query_batch = tuple(t.to(self.device) for t in query_batch)
                    q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                    q_outputs = fast_model(q_input_ids, q_attention_mask, q_segment_ids, labels = q_label_id)

                    q_logits = F.softmax(q_outputs[1], dim=1)
                    pre_label_id = torch.argmax(q_logits, dim=1)
                    pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                    q_label_id = q_label_id.detach().cpu().numpy().tolist()

                    acc = accuracy_score(pre_label_id, q_label_id)
                    task_accs.append(acc)
                    
                fast_model = fast_model.to(torch.device('cpu'))
                del fast_model, inner_optimizer, query_batch
                torch.cuda.empty_cache()
            
            if training:
                # Average gradient across tasks
                for i in range(0, len(sum_gradients)):
                    sum_gradients[i] = sum_gradients[i] / float(num_task)

                # Assign gradient for original model, then using optimizer to update its weights
                for i, params in enumerate(self.model.model.parameters()):
                    params.grad = sum_gradients[i]

                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()
                
                del sum_gradients
                gc.collect()
            
            return np.mean(task_accs)
        
        elif not training:

            for task_id, task in enumerate(batch_tasks):    
                support = task[0]
                query   = task[1]
                
                fast_model = deepcopy(self.model.model)
                fast_model = fast_model.to(self.device)
                support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                                batch_size=self.inner_batch_size)
                
                inner_optimizer = Adam(fast_model.parameters(), lr=self.inner_update_lr)
                fast_model.train()
                
                print('----Task', task_id, '----')

                best_val_acc = 0

                for i in range(0, num_inner_update_step):   
                    all_loss = []
                    for inner_step, batch in enumerate(support_dataloader):     
                        
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, segment_ids, label_id = batch
                        outputs = fast_model(input_ids, attention_mask, segment_ids, labels = label_id)
                        
                        loss = outputs[0]              
                        loss.backward()
                        inner_optimizer.step()
                        inner_optimizer.zero_grad()
                        
                        all_loss.append(loss.item()) 
                    
                    if i % 4 == 0:
                        print("Inner Loss: ", np.mean(all_loss))

                        fast_model.eval()

                        with torch.no_grad():
                            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
                            query_batch = iter(query_dataloader).next()
                            query_batch = tuple(t.to(self.device) for t in query_batch)
                            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                            q_outputs = fast_model(q_input_ids, q_attention_mask, q_segment_ids, labels = q_label_id)

                            q_logits = F.softmax(q_outputs[1], dim=1)
                            pre_label_id = torch.argmax(q_logits, dim=1)
                            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                            q_label_id = q_label_id.detach().cpu().numpy().tolist()

                            acc = accuracy_score(pre_label_id, q_label_id)
                            task_accs.append(acc)   
                                
                            if best_val_acc <  acc:
                                best_val_acc = acc
                                best_output_dir=f'../../../meta_learning_output/category_0/best_val_epoch-{self.epoch}_global_step_{self.all_step}_step-{self.outer_step}_inner_update_step-{i}'
                                self.model.save_model(output_dir=best_output_dir, model=fast_model)

            return best_val_acc, best_output_dir
                

if __name__ == "__main__":

    debug = True

    train_examples = json.load(open('../../../dataset/source_dataset/Reaxys-MultiCondi-Yield/meta_training_task_dataset.json'))
    test_spt_examples = json.load(open('../../../dataset/source_dataset/Reaxys-MultiCondi-Yield/meta_testing_support_set_category_0.json'))     # category: 0, 6, 9, 10, 11
    test_qry_examples = json.load(open('../../../dataset/source_dataset/Reaxys-MultiCondi-Yield/meta_testing_query_set_category_0.json'))

    if debug:
        test_spt_examples = test_spt_examples[:100]
        test_qry_examples = test_qry_examples[:20]

    print(len(train_examples), len(test_spt_examples), len(test_qry_examples))

    model_path = '../../../outputs/egret_pretrain_stage_2/bestoutput'
    args = TrainingArgs()
    learner = Learner(args)
    tokenizer = learner.model.tokenizer

    random_seed(123)
    test = MetaTestingTask(test_spt_examples, test_qry_examples, num_task=1, k_support=17544, k_query=2433, tokenizer = tokenizer)

    # start training
    global_step = 0
    meta_test_q_acc = 0     

    for epoch in range(args.meta_epoch):
        learner.epoch = epoch
        
        train = MetaTask(train_examples, num_task=100, k_support=80, k_query=20, tokenizer=tokenizer)
        db = create_batch_of_tasks(train, is_shuffle=True, batch_size=args.outer_batch_size)

        for step, task_batch in enumerate(db):
            learner.outer_step = step
            learner.all_step = global_step
            
            f = open('log.txt', 'a')
            training_acc = learner(task_batch)
            print('Step: ', step, '\ttraining Acc: ', training_acc)
            f.write('Step-' + str(step) + ' Train-acc: ' + str(training_acc) + '\n')
            
            if global_step % 20 == 0:
                random_seed(123)
                print("\n-----------------Testing Mode-----------------\n")
                db_test = create_batch_of_tasks(test, is_shuffle = False, batch_size = 1)
        
                for test_batch in db_test:
                    test_acc, val_best_model_dir = learner(test_batch, training = False)

                print('Step: ', step, 'Test F1: ', test_acc)
                f.write('Step-' + str(step) + ' Test-acc: ' + str(test_acc) + '\n')
                if meta_test_q_acc < test_acc:
                    meta_test_q_acc = test_acc
                    copy_model_state_file(val_best_model_dir, '../../../outputs/meta_learning_output/category_0/best_model')
            
            global_step += 1
            f.close()