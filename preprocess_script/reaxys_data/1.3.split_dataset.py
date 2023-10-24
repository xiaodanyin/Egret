
from random import shuffle
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def shuffle_data(ls):
    random.shuffle(ls)
    return ls

def split_train_val_test(ls):
    if len(ls) == 3:
        split_1 = int(1/3 * len(ls))
        split_2 = int(2/3 * len(ls))
        test_idx = ls[:split_1]
        val_idx = ls[split_1:split_2]
        train_idx = ls[split_2:]
    else:
        split_1 = int(0.13 * len(ls))
        split_2 = int(0.2 * len(ls))
        test_idx = ls[:split_1]
        val_idx = ls[split_1:split_2]
        train_idx = ls[split_2:]

    return train_idx, val_idx, test_idx

def split_trn_val_test(ls):

    split_1 = int(0.1 * len(ls))
    split_2 = int(0.2 * len(ls))
    test_id = ls[:split_1]
    val_id = ls[split_1:split_2]
    train_id = ls[split_2:]

    return train_id, val_id, test_id


if __name__ == '__main__':
    import random
    random.seed(123)

    df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/Reaxys_MultiCondi_Yield.csv', encoding='utf-8')
    rct_id_num = defaultdict(int)
    for rct_id in tqdm(df['Reaction ID'].tolist()):
        rct_id_num[rct_id] += 1
    rct_id_num_ls = list(rct_id_num.items())
    rct_id_num_ls.sort(key=lambda x:x[1], reverse = True)

    num_3_up = [k for k, v in rct_id_num_ls if v > 3]
    print(len(num_3_up)) 

    num_3 = [k for k, v in rct_id_num_ls if v == 3]
    print(len(num_3))

    num_2 = [k for k, v in rct_id_num_ls if v == 2]
    print(len(num_2)) 

    num_2_blow = [k for k, v in rct_id_num_ls if v < 2]
    print(len(num_2_blow)) 

    train_df = pd.DataFrame(columns = ['Reaction ID', 'Reaction', 'can_rxn_smiles', 'Reaction Type', 'new_time', 'new_temperature', 'new_reagent', 'new_catalyst', 
                                       'new_solvent', 'mapped_rxn_smiles', 'Links to Reaxys', 'Yield (numerical)', 'yield_difference', 'new_reagent_smi', 
                                       'new_catalyst_smi', 'new_solvent_smi', 'can_reagent_smi', 'can_catalyst_smi', 'can_solvent_smi'])
    val_df = pd.DataFrame(columns = ['Reaction ID', 'Reaction', 'can_rxn_smiles', 'Reaction Type', 'new_time', 'new_temperature', 'new_reagent', 'new_catalyst', 
                                       'new_solvent', 'mapped_rxn_smiles', 'Links to Reaxys', 'Yield (numerical)', 'yield_difference', 'new_reagent_smi', 
                                       'new_catalyst_smi', 'new_solvent_smi', 'can_reagent_smi', 'can_catalyst_smi', 'can_solvent_smi'])
    test_df = pd.DataFrame(columns = ['Reaction ID', 'Reaction', 'can_rxn_smiles', 'Reaction Type', 'new_time', 'new_temperature', 'new_reagent', 'new_catalyst', 
                                       'new_solvent', 'mapped_rxn_smiles', 'Links to Reaxys', 'Yield (numerical)', 'yield_difference', 'new_reagent_smi', 
                                       'new_catalyst_smi', 'new_solvent_smi', 'can_reagent_smi', 'can_catalyst_smi', 'can_solvent_smi'])

    for reaction_id in tqdm(num_3_up, total=len(num_3_up)):
        rct_id_df = df.loc[df['Reaction ID'] == reaction_id]
        rct_id_df = rct_id_df.reset_index(drop=True)

        idx_ls = []
        for idx, id in enumerate(rct_id_df['Reaction ID'].tolist()):
            idx_ls.append(idx)
        idx_ls = shuffle_data(idx_ls)

        train_idx, val_idx, test_idx = split_train_val_test(idx_ls)
        rct_id_train_df = rct_id_df.loc[train_idx]
        rct_id_val_df = rct_id_df.loc[val_idx]
        rct_id_test_df = rct_id_df.loc[test_idx]

        train_df = train_df.append(rct_id_train_df)
        val_df = val_df.append(rct_id_val_df)
        test_df = test_df.append(rct_id_test_df)

    for reaction_id in tqdm(num_3, total=len(num_3)):
        rct_id_df_ = df.loc[df['Reaction ID'] == reaction_id]
        rct_id_df_ = rct_id_df_.reset_index(drop=True)

        idx_ls = []
        for idx, id in enumerate(rct_id_df_['Reaction ID'].tolist()):
            idx_ls.append(idx)
        idx_ls = shuffle_data(idx_ls)

        _train_idx, _val_idx, _test_idx = split_train_val_test(idx_ls)       
        _rct_id_train_df = rct_id_df_.loc[_train_idx]
        _rct_id_val_df = rct_id_df_.loc[_val_idx]
        _rct_id_test_df = rct_id_df_.loc[_test_idx]
    
        train_df = train_df.append(_rct_id_train_df)
        val_df = val_df.append(_rct_id_val_df)
        test_df = test_df.append(_rct_id_test_df)

    shuffle_num_2 = shuffle_data(num_2)
    trn_id, val_id, test_id = split_trn_val_test(shuffle_num_2)

    trn_idx_2 = []
    val_idx_2 = []
    test_idx_2 = []

    for idx, reaction_id in enumerate(df['Reaction ID'].tolist()):
        if reaction_id in trn_id:
            trn_idx_2.append(idx)
        elif reaction_id in val_id:
            val_idx_2.append(idx)
        elif reaction_id in test_id:
            test_idx_2.append(idx)
    
    num_2_train_df = df.loc[trn_idx_2]
    num_2_val_df = df.loc[val_idx_2]
    num_2_test_df = df.loc[test_idx_2]

    train_df = train_df.append(num_2_train_df)
    val_df = val_df.append(num_2_val_df)
    test_df = test_df.append(num_2_test_df)

    train_df.to_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/trn_dataset.csv', encoding='utf-8', index=False)
    val_df.to_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/val_dataset.csv', encoding='utf-8', index=False)
    test_df.to_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/test_dataset.csv', encoding='utf-8', index=False)










