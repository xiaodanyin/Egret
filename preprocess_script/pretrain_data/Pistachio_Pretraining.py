import pandas as pd
import os
from tqdm import tqdm
from rdkit import Chem
from rxnmapper import RXNMapper
import re


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

def process_reaction(rxn):
    reactants, reagents, products = rxn.split(">")
    try:
        precursors = [canonicalize_smi(r, True) for r in reactants.split(".")]
        if len(reagents) > 0:
            reagents = [canonicalize_smi(r, True) for r in reagents.split(".")]
        else:
            reagents = ['']
        products = [canonicalize_smi(p, True) for p in products.split(".")]
    except NotCanonicalizableSmilesException:
        return "", ''
    joined_precursors = ".".join(sorted(precursors))
    joined_products = ".".join(sorted(products))
    joined_reagents = '.'.join(sorted(reagents))
    return f"{joined_precursors}>>{joined_products}", joined_reagents

def process_reaction_(rxn, reagent):
    reactants, products = rxn.split(">>")
    try:
        precursors = [canonicalize_smi(r, False) for r in reactants.split(".")]
        if not pd.isna(reagent):
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

def process_reagent(rgt_smi):
    mol = Chem.MolFromSmiles(rgt_smi)
    if not mol:
        return ''
    else:
        return Chem.MolToSmiles(mol)

if __name__ == '__main__':
    pistachio_df = pd.read_csv('../../dataset/pretrain_data/pistachio_pretrain/9321535_add_year.csv', header=None)
    pistachio_df.columns = ['rxn_smiles', 'id', 'rxn_class_id', 'rxn_class', 'unk', 'years']
    pistachio_df.drop_duplicates(subset=['rxn_smiles'], keep='first', inplace=True)

    canonical_rxn_smiles_ls = []
    org_reagent_ls = []
    for rxn_smi in tqdm(pistachio_df['rxn_smiles'].to_list()):
        canonical_rxn_smi, can_reagents= process_reaction(rxn_smi)
        canonical_rxn_smiles_ls.append(canonical_rxn_smi)
        org_reagent_ls.append(can_reagents)
    assert len(canonical_rxn_smiles_ls) == len(pistachio_df)
    assert len(org_reagent_ls) == len(pistachio_df)
    pistachio_df['canonical_rxn_smiles'] = canonical_rxn_smiles_ls
    pistachio_df['org_rxn_can_reagents'] = org_reagent_ls
    pistachio_df = pistachio_df.loc[pistachio_df['canonical_rxn_smiles'] != '']
    pistachio_df.drop_duplicates(subset=['canonical_rxn_smiles', 'org_rxn_can_reagents'], keep='first', inplace=True)
    
    rxn_mapper = RXNMapper()
    mapped_can_rxn_smiles_ls = []
    confidence_ls = []
    for can_rxn_smi in tqdm(pistachio_df['canonical_rxn_smiles'].to_list()):
        try:
            results = rxn_mapper.get_attention_guided_atom_maps([can_rxn_smi])[0]
            mapped_can_rxn_smiles_ls.append(results['mapped_rxn'])
            confidence_ls.append(results['confidence'])
        except:
            mapped_can_rxn_smiles_ls.append('')
            confidence_ls.append('')
    assert len(mapped_can_rxn_smiles_ls) == len(pistachio_df)
    pistachio_df['mapped_can_rxn_smiles'] = mapped_can_rxn_smiles_ls
    pistachio_df['confidence'] = confidence_ls
    pistachio_df = pistachio_df.loc[pistachio_df['mapped_can_rxn_smiles'] != '']
    
    mapped_can_rxn_smiles_no_reagent_ls = []
    identify_reagent_ls = []
    for mapped_can_rxn_smi in tqdm(pistachio_df['mapped_can_rxn_smiles'].to_list()):
        new_precursors, new_products = mapped_can_rxn_smi.split('>>')
        pt = re.compile(r':(\d+)]')
        new_reac_ls = []
        new_reag_ls = []
        new_precursors_ls = new_precursors.split('.')
        for precursor in new_precursors_ls:
            if re.findall(pt, precursor):
                new_reac_ls.append(precursor)
            else:
                new_reag_ls.append(precursor)
        if new_reag_ls:
            if len(new_reag_ls) > 1:
                identify_reagent_ls.append('.'.join(new_reag_ls))
            else:
                identify_reagent_ls.append(new_reag_ls[0])
        else:
            identify_reagent_ls.append('')
        joined_new_precursors = ".".join(sorted(new_reac_ls))
        mapped_can_rxn_smiles_no_reagent_ls.append(f'{joined_new_precursors}>>{new_products}')          
    assert len(mapped_can_rxn_smiles_no_reagent_ls) == len(pistachio_df)
    assert len(identify_reagent_ls) == len(pistachio_df)
    pistachio_df['mapped_can_rxn_smiles_no_reagent'] = mapped_can_rxn_smiles_no_reagent_ls
    pistachio_df['identify_reagents'] = identify_reagent_ls
    
    fin_can_rxn_smi_ls = []
    identify_can_rgt_ls = []
    for rxn_smi_no_rgt in tqdm(pistachio_df['mapped_can_rxn_smiles_no_reagent'].to_list()):
        fin_can_rxn_smi, _ = process_reaction(rxn_smi_no_rgt)
        fin_can_rxn_smi_ls.append(fin_can_rxn_smi)
    for rgt in tqdm(pistachio_df['identify_reagents'].to_list()):
        fin_reagent = process_reagent(rgt)
        identify_can_rgt_ls.append(fin_reagent)
    assert len(fin_can_rxn_smi_ls) == len(pistachio_df)
    assert len(identify_can_rgt_ls) == len(pistachio_df)
    pistachio_df['identify_can_reagents'] = identify_can_rgt_ls
    pistachio_df['final_can_rxn_smiles'] = fin_can_rxn_smi_ls
    pistachio_df = pistachio_df.loc[pistachio_df['final_can_rxn_smiles'] != '']
   
    final_can_regent_ls = []
    for org_rgt, itf_rgt in tqdm(zip(pistachio_df['org_rxn_can_reagents'].to_list(), pistachio_df['identify_can_reagents'].to_list())):
        if org_rgt != '' and itf_rgt != '':
            rgt_ls = [org_rgt, itf_rgt]
            joined_rgt = '.'.join(sorted(rgt_ls))
            final_can_regent_ls.append(joined_rgt)
        elif org_rgt != '' and itf_rgt == '':      
            final_can_regent_ls.append(org_rgt)
        elif org_rgt == '' and itf_rgt != '':      
            final_can_regent_ls.append(itf_rgt)
        elif org_rgt == '' and itf_rgt == '':      
            final_can_regent_ls.append('')
    assert len(final_can_regent_ls) == len(pistachio_df)
    pistachio_df['final_can_regent'] = final_can_regent_ls
    
    fin_rxn_smiles_ls = []
    for fin_rxn_smi, fin_rgt in tqdm(zip(fin_can_rxn_smi_ls, final_can_regent_ls), total=len(pistachio_df)):
        fin_rxn_smiles = process_reaction_(fin_rxn_smi, fin_rgt)
        fin_rxn_smiles_ls.append(fin_rxn_smiles)   
    pistachio_df['final_canonical_rxn_smiles'] = fin_rxn_smiles_ls 
    pistachio_df = pistachio_df.loc[pistachio_df['final_canonical_rxn_smiles'] != '']    

    pistachio_pretraining = pistachio_df.sample(frac=1, random_state=123)
    pistachio_pretraining.to_csv('../../dataset/pretrain_data/pistachio_pretraining.csv', index=False, encoding='utf-8')
    rxn_smi_ls = pistachio_pretraining['final_canonical_rxn_smiles'].to_list()
    pistachio_pretraining_train = rxn_smi_ls[ :1833798]
    pistachio_pretraining_eval = rxn_smi_ls[1833798: ]
    with open(os.path.join('../../dataset/pretrain_data/pistachio_pretraining_eval.txt'), 'w') as f1:
        for eval_smi in tqdm(pistachio_pretraining_eval, total=len(pistachio_pretraining_eval)):
            f1.write(eval_smi + '\n')
    with open(os.path.join('../../dataset/pretrain_data/pistachio_pretraining_train.txt'), 'w') as f2:
        for train_smi in tqdm(pistachio_pretraining_train, total=len(pistachio_pretraining_train)):
            f2.write(train_smi + '\n')    
        