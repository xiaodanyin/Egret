import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import json
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

class NotCanonicalizableSmilesException(ValueError):
    pass

def canonicalize_smi(smi, remove_atom_mapping=False):
    r"""
    Canonicalize SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise NotCanonicalizableSmilesException("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)

def process_reaction(rxn, reagent):
    """
    Process and canonicalize reaction SMILES 
    """
    reactants, products = rxn.split(">>")
    try:
        precursors = [canonicalize_smi(r, False) for r in reactants.split(".")]
        # if not pd.isna(reagent):
        if reagent != '':
            precursors += [
                canonicalize_smi(r, False) for r in reagent.split(".")
            ]
        else:
            pass
        products = [canonicalize_smi(p, False) for p in products.split(".")]
    except NotCanonicalizableSmilesException:
        return ""

    joined_precursors = ".".join(sorted(precursors))
    joined_products = ".".join(sorted(products))
    return f"{joined_precursors}>>{joined_products}"

if __name__ == "__main__":
    import random
    random.seed(123) 

    pistachio_df = pd.read_csv('../../dataset/pretrain_data/pistachio_reaction_condition_identify.csv', encoding='utf-8')
    with open('../../dataset/pretrain_data/pistachio_condition_dict.json', encoding='utf-8') as f:
        condition_dict = json.load(f)

    catalyst_ls = condition_dict['catalyst']
    reagent_ls = condition_dict['reagent']
    solvent_ls = condition_dict['solvent']
    unknown_ls = condition_dict['unknown']
 
    negative_data = {
        'ID': [],
        'rxn_smi': [],
        'catalyst': [],
        'reagent': [],
        'solvent': [],
        'unknown': []
    }
    for (id, canonical_rxn, cat, rea, sol, unk) in zip(pistachio_df['_ID'].tolist(),
                                                   pistachio_df['final_can_rxn_smiles'].tolist(), 
                                                   pistachio_df['catalyst'].tolist(), 
                                                   pistachio_df['reagent'].tolist(), 
                                                   pistachio_df['solvent'].tolist(), 
                                                   pistachio_df['unknown'].tolist()):
        
        negative_data['ID'].append(id)
        negative_data['rxn_smi'].append(canonical_rxn)
        if not pd.isna(cat):
            cat_ = cat.split(';')
            if len(cat_) != 1:
                cat_ = '.'.join(cat_)
                negative_data['catalyst'].append(cat_)
            else:
                negative_data['catalyst'].append(cat_[0])
        else:
            negative_data['catalyst'].append('')
        if not pd.isna(rea):
            rea_ = rea.split(';')
            if len(rea_) != 1:
                rea_ = '.'.join(rea_)
                negative_data['reagent'].append(rea_)
            else:
                negative_data['reagent'].append(rea_[0])
        else:
            negative_data['reagent'].append('')
        if not pd.isna(sol):
            sol_ = sol.split(';')
            if len(sol_) != 1:
                sol_ = '.'.join(sol_)
                negative_data['solvent'].append(sol_)
            else:
                negative_data['solvent'].append(sol_[0])
        else:
            negative_data['solvent'].append('')
        if not pd.isna(unk):
            unk_ = unk.split(';')
            if len(unk_) != 1:
                unk_ = '.'.join(unk_)
                negative_data['unknown'].append(unk_)
            else:
                negative_data['unknown'].append(unk_[0])
        else:
            negative_data['unknown'].append('')

        count_list = list(range(0, 4))
        for i in count_list:
            negative_data['ID'].append(id)
            negative_data['rxn_smi'].append(canonical_rxn)
            if not pd.isna(cat):
                cat_ls = cat.split(';')
                cat_len = len(cat_ls)
                random_cat = random.sample(catalyst_ls, cat_len) 
                if cat_len != 1:
                    random_cat = '.'.join(random_cat)
                    negative_data['catalyst'].append(random_cat)
                else:
                    negative_data['catalyst'].append(random_cat[0])
            else:
                random_cat = random.sample(catalyst_ls, 1) 
                negative_data['catalyst'].append(random_cat[0])
            if not pd.isna(rea):
                rea_ls = rea.split(';')
                rea_len = len(rea_ls)
                random_rea = random.sample(reagent_ls, rea_len) 
                if rea_len != 1:
                    random_rea = '.'.join(random_rea)
                    negative_data['reagent'].append(random_rea)
                else:
                    negative_data['reagent'].append(random_rea[0])
            else:
                random_rea = random.sample(reagent_ls, 1)
                negative_data['reagent'].append(random_rea[0])
            if not pd.isna(sol):  
                sol_ls = sol.split(';')
                sol_len = len(sol_ls)
                random_sol = random.sample(solvent_ls, sol_len)
                if sol_len != 1:
                    random_sol = '.'.join(random_sol)
                    negative_data['solvent'].append(random_sol)
                else:
                    negative_data['solvent'].append(random_sol[0])
            else:
                random_sol = random.sample(solvent_ls, 1)
                negative_data['solvent'].append(random_sol[0])
            if not pd.isna(unk):
                unk_ls = unk.split(';')
                unk_len = len(unk_ls)
                random_unk = random.sample(unknown_ls, unk_len)
                if unk_len != 1:
                    random_unk = '.'.join(random_unk)
                    negative_data['unknown'].append(random_unk)
                else:
                    negative_data['unknown'].append(random_unk[0])
            else:
                negative_data['unknown'].append('')
    assert len(negative_data['ID']) == len(negative_data['rxn_smi'])
    assert len(negative_data['ID']) == len(negative_data['catalyst'])
    assert len(negative_data['ID']) == len(negative_data['reagent'])
    assert len(negative_data['ID']) == len(negative_data['solvent'])
    assert len(negative_data['ID']) == len(negative_data['unknown'])
    condition_aug_df = pd.DataFrame.from_dict(negative_data)

    condition_merge = []
    for cat_, rea_, sol_, unk_ in zip(condition_aug_df['catalyst'].tolist(),
                                      condition_aug_df['reagent'].tolist(),
                                      condition_aug_df['solvent'].tolist(),
                                      condition_aug_df['unknown'].tolist()):
        cat_s = cat_.split('.')
        rea_s = rea_.split('.')
        sol_s = sol_.split('.')
        unk_s = unk_.split('.')
        cat_s_ = [ x for x in cat_s if x != '']
        rea_s_ = [ x for x in rea_s if x != ''] 
        sol_s_s = [ x for x in sol_s if x != ''] 
        unk_s_s = [ x for x in unk_s if x != ''] 
        condition_ls = cat_s_ + rea_s_ + sol_s_s + unk_s_s
        conditions = '.'.join(condition_ls)
        condition_merge.append(conditions)

    condition_aug_df['merge_condition'] = condition_merge
    assert len(condition_merge) == len(condition_aug_df)

    final_rxn_smi_ls = []
    for rxn_smi, conditon in tqdm(
                                 zip(condition_aug_df['rxn_smi'].tolist(), 
                                 condition_aug_df['merge_condition'].tolist()), 
                                 total=len(condition_aug_df)
    ):
        fin_rxn_smiles = process_reaction(rxn_smi, conditon)
        final_rxn_smi_ls.append(fin_rxn_smiles)
    condition_aug_df['final_rxn_smi'] = final_rxn_smi_ls
    condition_aug_df = condition_aug_df.loc[condition_aug_df['final_rxn_smi'] != '']
    print(len(condition_aug_df))
    condition_aug_df.to_csv('../../dataset/pretrain_data/pistachio_negative_sample_construction.csv', encoding='utf-8', index=False)
