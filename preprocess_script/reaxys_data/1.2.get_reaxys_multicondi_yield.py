import pandas as pd
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from rdkit import Chem


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return ''
    else:
        return Chem.MolToSmiles(mol)
    
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

def process_reaction(rxn, condi_smi):
    """
    Process and canonicalize reaction SMILES 
    """
    reactants, products = rxn.split(">>")
    try:
        precursors = [canonicalize_smi(r, False) for r in reactants.split(".")]
        if condi_smi != '':
            precursors += [
                canonicalize_smi(r, False) for r in condi_smi.split("; ")
            ]
        else:
            pass
        products = [canonicalize_smi(p, False) for p in products.split(".")]
    except NotCanonicalizableSmilesException:
        return ""
    joined_precursors = ".".join(sorted(precursors))
    joined_products = ".".join(sorted(products))
    return f"{joined_precursors}>>{joined_products}"

    

if __name__ == '__main__':
    reaxys_yield_diffence_20_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/total_syn_multi_yield_difference_20.csv', encoding='utf-8')
    reaxys_con_name2smiles_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/reaxys_condition_names2smiles.csv', encoding='utf-8')

    condition_name2smiles = {}
    for name, smiles in tqdm(zip(reaxys_con_name2smiles_df['name'].tolist(), reaxys_con_name2smiles_df['smiles'].tolist()), total=len(reaxys_con_name2smiles_df)):
        if not pd.isna(smiles):
            condition_name2smiles[name] = smiles
    cannot_convert_ixd = []
    cannot_all_convert_idx = []
    rgt_smiles_ls = []
    cat_smiles_ls = []
    sol_smiles_ls = []
    for i, (reagent, catalyst, solvent) in tqdm(enumerate(zip(reaxys_yield_diffence_20_df['new_reagent'].tolist(), 
                                                              reaxys_yield_diffence_20_df['new_catalyst'].tolist(), 
                                                              reaxys_yield_diffence_20_df['new_solvent'].tolist())), 
                                                              total=len(reaxys_yield_diffence_20_df)):
        if pd.isna(reagent):
            rgt_smiles_ls.append('')
        else:
            new_rgt = reagent.split('; ')
            new_rgt_smi = []
            cannot_convert_num = 0
            for rgt in new_rgt:
                if rgt in condition_name2smiles:
                    new_rgt_smi.append(condition_name2smiles[rgt])
                else:
                    cannot_convert_num += 1
                    continue
            if cannot_convert_num != 0 and cannot_convert_num != len(new_rgt):
                cannot_all_convert_idx.append(i)
                new_rgt_smi = '; '.join(new_rgt_smi)
                rgt_smiles_ls.append(new_rgt_smi)
            elif cannot_convert_num == len(new_rgt):
                cannot_convert_ixd.append(i)
                new_rgt_smi = '; '.join(new_rgt_smi)
                rgt_smiles_ls.append(new_rgt_smi)
            else:
                new_rgt_smi = '; '.join(new_rgt_smi)
                rgt_smiles_ls.append(new_rgt_smi)
        if pd.isna(catalyst):
                cat_smiles_ls.append('')
        else:
            new_cat = catalyst.split('; ')
            new_cat_smi = []
            cannot_convert_num = 0
            for cat in new_cat:
                if cat in condition_name2smiles:
                    new_cat_smi.append(condition_name2smiles[cat])
                else:
                    cannot_convert_num += 1
                    continue
            if cannot_convert_num != 0 and cannot_convert_num != len(new_cat):
                cannot_all_convert_idx.append(i)
                new_cat_smi = '; '.join(new_cat_smi)
                cat_smiles_ls.append(new_cat_smi)
            elif cannot_convert_num == len(new_cat):   
                cannot_convert_ixd.append(i)
                new_cat_smi = '; '.join(new_cat_smi)
                cat_smiles_ls.append(new_cat_smi)
            else:
                new_cat_smi = '; '.join(new_cat_smi)
                cat_smiles_ls.append(new_cat_smi)
        if pd.isna(solvent):
            sol_smiles_ls.append('')
        else:
            new_sol = solvent.split('; ')
            new_sol_smi = []
            cannot_convert_num = 0
            for sol in new_sol:
                if sol in condition_name2smiles:
                    new_sol_smi.append(condition_name2smiles[sol])
                else:
                    cannot_convert_num += 1
                    continue
            if cannot_convert_num != 0 and cannot_convert_num != len(new_sol):
                cannot_all_convert_idx.append(i)
                new_sol_smi = '; '.join(new_sol_smi)
                sol_smiles_ls.append(new_sol_smi)
            elif cannot_convert_num == len(new_sol):
                cannot_convert_ixd.append(i)
                new_sol_smi = '; '.join(new_sol_smi)
                sol_smiles_ls.append(new_sol_smi)
            else:
                new_sol_smi = '; '.join(new_sol_smi)
                sol_smiles_ls.append(new_sol_smi)
    reaxys_yield_diffence_20_df['new_reagent_smi'] = rgt_smiles_ls
    reaxys_yield_diffence_20_df['new_catalyst_smi'] = cat_smiles_ls
    reaxys_yield_diffence_20_df['new_solvent_smi'] = sol_smiles_ls
    cannot_convert_ixd.extend(cannot_all_convert_idx)
    
    select_idx = []
    for idx in tqdm(range(len(reaxys_yield_diffence_20_df)), total=len(reaxys_yield_diffence_20_df)):
        if idx in set(cannot_convert_ixd):
            continue
        else:
            select_idx.append(idx)          
    reaxys_yield_diffence_20_df = reaxys_yield_diffence_20_df.loc[select_idx]
    # print(len(reaxys_yield_diffence_20_df))
    cannot_canonical_idx = []
    can_rgt_smiles_ls = []
    can_cat_smiles_ls = []
    can_sol_smiles_ls = []
    for i, (reagent, catalyst, solvent) in tqdm(enumerate(zip(reaxys_yield_diffence_20_df['new_reagent_smi'].tolist(), 
                                                reaxys_yield_diffence_20_df['new_catalyst_smi'].tolist(), 
                                                reaxys_yield_diffence_20_df['new_solvent_smi'].tolist())), 
                                                total=len(reaxys_yield_diffence_20_df)):
                                                if pd.isna(reagent):
                                                    can_rgt_smiles_ls.append('')
                                                else:
                                                    can_rgts_ls = []
                                                    rgts_ls = reagent.split('; ')
                                                    for rgt in rgts_ls:
                                                        can_rgt = canonicalize_smiles(rgt)
                                                        can_rgts_ls.append(can_rgt)
                                                    if len(can_rgts_ls) != len(rgts_ls):
                                                        cannot_canonical_idx.append(i)
                                                    else:
                                                        can_rgts = '; '.join(can_rgts_ls)
                                                        can_rgt_smiles_ls.append(can_rgts)
                                                if pd.isna(catalyst):
                                                    can_cat_smiles_ls.append('')
                                                else:
                                                    can_cats_ls = []
                                                    cats_ls = catalyst.split('; ')
                                                    for cat in cats_ls:
                                                        can_cat = canonicalize_smiles(cat)
                                                        can_cats_ls.append(can_cat)
                                                    if len(can_cats_ls) != len(cats_ls):
                                                        cannot_canonical_idx.append(i)
                                                    else:
                                                        can_cats = '; '.join(can_cats_ls)
                                                        can_cat_smiles_ls.append(can_cats)
                                                if pd.isna(solvent):
                                                    can_sol_smiles_ls.append('')
                                                else:
                                                    can_sols_ls = []
                                                    sols_ls = solvent.split('; ')
                                                    for sol in sols_ls:
                                                        can_sol = canonicalize_smiles(sol)
                                                        can_sols_ls.append(can_sol)                                                   
                                                    if len(can_sols_ls) != len(sols_ls):
                                                        cannot_canonical_idx.append(i)
                                                    else:
                                                        can_sols = '; '.join(can_sols_ls)
                                                        can_sol_smiles_ls.append(can_sols)
    assert len(can_rgt_smiles_ls) == len(can_cat_smiles_ls) 
    assert len(can_sol_smiles_ls) == len(can_cat_smiles_ls)
    # print(len(cannot_canonical_idx))
    reaxys_yield_diffence_20_df['can_reagent_smi'] = can_rgt_smiles_ls
    reaxys_yield_diffence_20_df['can_catalyst_smi'] = can_cat_smiles_ls
    reaxys_yield_diffence_20_df['can_solvent_smi'] = can_sol_smiles_ls
    select_idx_ = []
    for idx in tqdm(range(len(reaxys_yield_diffence_20_df)), total=len(reaxys_yield_diffence_20_df)):
        if idx in set(cannot_canonical_idx):
            continue
        else:
            select_idx_.append(idx)         
    reaxys_yield_diffence_20_df = reaxys_yield_diffence_20_df.loc[select_idx_].reset_index(inplace=True)
    print(len(reaxys_yield_diffence_20_df))   
    print(len(set(reaxys_yield_diffence_20_df['Reaction ID'].tolist())))    
    print(len(set(reaxys_yield_diffence_20_df['Reaction Type'].tolist())))   
    print(len(set(reaxys_yield_diffence_20_df['can_reagent_smi'].tolist())))      
    print(len(set(reaxys_yield_diffence_20_df['can_catalyst_smi'].tolist())))   
    print(len(set(reaxys_yield_diffence_20_df['can_solvent_smi'].tolist())))    

    condition_ls = []
    for rgt, cat, sol in tqdm(zip(reaxys_yield_diffence_20_df['can_reagent_smi'].tolist(), 
                             reaxys_yield_diffence_20_df['can_catalyst_smi'].tolist(),
                             reaxys_yield_diffence_20_df['can_solvent_smi'].tolist()), total=len(reaxys_yield_diffence_20_df)):
                             condition = []
                             if not pd.isna(rgt):
                                 condition.append(rgt)
                             if not pd.isna(cat):
                                 condition.append(cat)
                             if not pd.isna(sol):
                                 condition.append(sol)                        
                             if len(condition) == 0:
                                condition_ls.append('')
                             else:
                                conditions = '; '.join(condition)
                                condition_ls.append(conditions)
    assert len(condition_ls) == len(reaxys_yield_diffence_20_df)
    reaxys_yield_diffence_20_df['conditions_smi'] = condition_ls

    fin_rxn_smiles_ls = []
    for can_rxn_smi, condi_smi in tqdm(zip(reaxys_yield_diffence_20_df['can_rxn_smiles'].tolist(), reaxys_yield_diffence_20_df['conditions_smi'].tolist()), total=len(reaxys_yield_diffence_20_df)):
      fin_rxn_smi = process_reaction(can_rxn_smi, condi_smi)
      fin_rxn_smiles_ls.append(fin_rxn_smi)
    assert len(fin_rxn_smiles_ls) == len(reaxys_yield_diffence_20_df)
    reaxys_yield_diffence_20_df['fin_can_rxn_smi'] = fin_rxn_smiles_ls    
    reaxys_yield_diffence_20_df.drop_duplicates(subset='fin_can_rxn_smi', keep='last', inplace=True)
    # print(len(reaxys_yield_diffence_20_df)) 
    rct_id_ls = []
    select_rct_id = []
    for i, rct_id in enumerate(reaxys_yield_diffence_20_df['Reaction ID'].tolist()):
        if rct_id in rct_id_ls:
            select_rct_id.append(rct_id)
        else:
            rct_id_ls.append(rct_id)
    select_rct_id = set(select_rct_id)
    select_id_idx = []
    for i, rct in enumerate(reaxys_yield_diffence_20_df['Reaction ID'].tolist()):
        if rct in select_rct_id:
            select_id_idx.append(i)
    final_reaxys_diffence20_df = reaxys_yield_diffence_20_df.loc[select_id_idx]
    print(len(final_reaxys_diffence20_df)) # 84125
    print(len(set(final_reaxys_diffence20_df['Reaction ID'].tolist()))) # 11831
    final_reaxys_diffence20_df.to_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/Reaxys_MultiCondi_Yield.csv', encoding='utf-8', index=False)                