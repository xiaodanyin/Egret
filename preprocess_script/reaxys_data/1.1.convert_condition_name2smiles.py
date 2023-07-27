from re import L
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from urllib.request import urlopen
from urllib.parse import quote
import os


try:
    import cirpy
except:
    pass
import json
import time
try:
    import sys
    sys.path.append('D:\software\ChemScript\Lib')
    from ChemScript16 import *
except:
    pass

def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return ''

if __name__ == '__main__':
    reaxys_yield_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/total_syn_multi_yield_difference_20.csv', encoding='utf-8')
    chemical_vocab = []
    
    reagent_ls = []
    for new_rgt in tqdm(reaxys_yield_df['new_reagent'].tolist(), total=len(reaxys_yield_df)):
        if not pd.isna(new_rgt):
            rgt_ls = new_rgt.split('; ')
            reagent_ls.extend(rgt_ls)
        else:
            continue
    reagent_dict = defaultdict(int)
    for rgt in tqdm(reagent_ls):
        reagent_dict[rgt] += 1
    reagents = list(reagent_dict.items())
    reagents.sort(key=lambda x:x[1], reverse=True)
    reagent = [k for(k,v) in reagents]
    chemical_vocab.extend(reagent)
    
    catalyst_ls = []
    for new_cat in tqdm(reaxys_yield_df['new_catalyst'].tolist(), total=len(reaxys_yield_df)):
        if not pd.isna(new_cat):
            cat_ls = new_cat.split('; ')
            catalyst_ls.extend(cat_ls)
        else:
            continue
    catalyst_dict = defaultdict(int)
    for cat in tqdm(catalyst_ls):
        catalyst_dict[cat] += 1
    catalysts = list(catalyst_dict.items())
    catalysts.sort(key=lambda x:x[1], reverse=True)
    catalyst = [k for(k,v) in catalysts]   
    chemical_vocab.extend(catalyst)

    solvent_ls = []
    for new_sol in tqdm(reaxys_yield_df['new_solvent'].tolist(), total=len(reaxys_yield_df)):
        if not pd.isna(new_sol):
            sol_ls = new_sol.split('; ')
            solvent_ls.extend(sol_ls)
        else:
            continue
    solvent_dict = defaultdict(int)
    for sol in tqdm(solvent_ls):
        solvent_dict[sol] += 1
    solvents = list(solvent_dict.items())
    solvents.sort(key=lambda x:x[1], reverse=True)
    solvent = [k for(k,v) in solvents]
    chemical_vocab.extend(solvent)
    chemical_vocab = set(chemical_vocab)
    print(len(chemical_vocab))

    conditionName2Smiles = {}
    chemdraw_cannot_convert = []
    for i, name in enumerate(chemical_vocab):
        if i % 100 == 0:
            print(i)
        if '||' in name:
            name = name.split('||')[0]
        try:
            smiles = cirpy.resolve(name, 'smiles')
        except:
            m = StructureData.LoadData(name)
            if hasattr(m, 'Smiles'):
                smiles = m.Smiles
            else:
                smiles = ''
        if smiles:
            conditionName2Smiles[name] = smiles
        else:
            conditionName2Smiles[name] = ''
            chemdraw_cannot_convert.append(name)
    f = open('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/search_chemdraw_cannot_convert.txt', 'w')
    for name in tqdm(chemdraw_cannot_convert):
        smiles = CIRconvert(name)
        f.write(('{}:{}'.format(name, smiles)) + '\n')
        f.flush()
    f.close()
    with open('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/search_chemdraw_cannot_convert.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        not_cvt_name_list = []
        for line in lines:
            try:
                name, smiles = line.split(':')
            except:
                continue
            if smiles == '\n':
              not_cvt_name_list.append(name) 
            else:
              conditionName2Smiles[name] = smiles.split('\n')[0]   
        ls = list(conditionName2Smiles.items())
        search_chemdraw_name2smiles_df = pd.DataFrame(ls)
        search_chemdraw_name2smiles_df.columns = ['name', 'smiles']    
        search_chemdraw_name2smiles_df.to_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/reaxys_condition_names2smiles.csv', encoding='utf-8', index=False)       