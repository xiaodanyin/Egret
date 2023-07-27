import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import json


if __name__ == "__main__":
    df = pd.read_csv('../../dataset/pretrain_data/pistachio_reaction_condition_identify.csv', encoding='utf-8')
    condition_dict = {
        'catalyst': [],
        'reagent': [],
        'solvent': [],
        'unknown': [],
    }
    for (cat, rea, sol, unk) in zip(df['catalyst'].tolist(), 
                                    df['reagent'].tolist(), 
                                    df['solvent'].tolist(), 
                                    df['unknown'].tolist()):
        if not pd.isna(cat):
            cat = cat.split(';')
            condition_dict['catalyst'].extend(cat)
        if not pd.isna(rea):
            rea = rea.split(';')
            condition_dict['reagent'].extend(rea)
        if not pd.isna(sol):
            sol = sol.split(';')
            condition_dict['solvent'].extend(sol)
        if not pd.isna(unk):
            unk = unk.split(";")
            condition_dict['unknown'].extend(unk)
    catalyst_ls = list(set(condition_dict['catalyst']))
    catalyst_ls.sort()
    reagent_ls = list(set(condition_dict['reagent']))
    reagent_ls.sort()
    solvent_ls = list(set(condition_dict['solvent']))
    solvent_ls.sort()
    unknown_ls = list(set(condition_dict['unknown']))
    unknown_ls.sort()
    condition_dict_1 = {
        'catalyst': catalyst_ls,
        'reagent': reagent_ls,
        'solvent': solvent_ls,
        'unknown': unknown_ls,
    }
    with open('../../dataset/pretrain_data/pistachio_condition_dict.json', 'w') as f:
         json.dump(condition_dict_1, f)