import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import random
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def get_random_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, doRandom=True)

def get_random_rxn(rxn):
    react, prod = rxn.split('>>')
    react_list = react.split('.')
    react_list = [get_random_smiles(smi) for smi in react_list]
    random.shuffle(react_list)
    new_react = '.'.join(react_list)
    prod_list = prod.split('.')
    prod_list = [get_random_smiles(smi) for smi in prod_list]
    random.shuffle(prod_list)
    new_prod = '.'.join(prod_list)
    return f'{new_react}>>{new_prod}'

def construct_positive_rxn_smiles(rxn_smi):
    results = []
    for i in range(4):
        results.append(get_random_rxn(rxn_smi))
    posi_smi_1, posi_smi_2, posi_smi_3, posi_smi_4 = results
    return posi_smi_1, posi_smi_2, posi_smi_3, posi_smi_4

if __name__ == '__main__':
    import random
    random.seed(123)
    df = pd.read_csv('../../dataset/pretrain_data/pistachio_negative_sample_construction.csv', encoding='utf-8')
    rxn_id_ls = list(set(df['ID'].tolist()))
    rxn_id_ls.sort()
    val_id_ls = random.sample(rxn_id_ls, 1000)

    df_dict = {
        'ID': [],
        'org_smi': [],
        'positive_smi_1': [],
        'positive_smi_2': [],
        'positive_smi_3': [],
        'positive_smi_4': [],
    }
    for rxn_id, final_rxn_smi in tqdm(zip(df['ID'].tolist(), df['final_rxn_smi'].tolist()), total=len(df)):
        posi_smi_1, posi_smi_2, posi_smi_3, posi_smi_4 = construct_positive_rxn_smiles(final_rxn_smi)
        df_dict['ID'].append(rxn_id)
        df_dict['org_smi'].append(final_rxn_smi)
        df_dict['positive_smi_1'].append(posi_smi_1)
        df_dict['positive_smi_2'].append(posi_smi_2)
        df_dict['positive_smi_3'].append(posi_smi_3)
        df_dict['positive_smi_4'].append(posi_smi_4)
    contrstive_df = pd.DataFrame.from_dict(df_dict)  
    train_idx = []
    val_idx = []
    for i, reaction_id in enumerate(contrstive_df['ID'].tolist()):
        if reaction_id in val_id_ls:
            val_idx.append(i)
        else:
            train_idx.append(i)
    assert len(train_idx) + len(val_idx) ==  len(contrstive_df)
    pistachio_contrasive_learning_train_df = contrstive_df.loc[train_idx]
    pistachio_contrasive_learning_eval_df = contrstive_df.loc[val_idx]
    contrstive_df.to_csv('../../dataset/pretrain_data/aug_pistachio_pretraining.csv', encoding='utf-8', index=False)
    pistachio_contrasive_learning_train_df.to_csv('../../dataset/pretrain_data/aug_pistachio_pretraining_train.csv', encoding='utf-8', index=False)
    pistachio_contrasive_learning_eval_df.to_csv('../../dataset/pretrain_data/aug_pistachio_pretraining_eval.csv', encoding='utf-8', index=False)    








