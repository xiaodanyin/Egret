import pandas as pd
import os
from rdkit import Chem
from tqdm import tqdm
from rxnmapper import RXNMapper
from collections import defaultdict
import numpy as np

def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''

def canonicalize_reaction(rxn_smiles):
    try:
        reactants, products = rxn_smiles.split('>>')
        canonicalize_reactants = canonicalize_smiles(reactants)
        canonicalize_products = canonicalize_smiles(products)
        if '' in [canonicalize_products, canonicalize_reactants]:
            return ''
        else:
            return f'{canonicalize_reactants}>>{canonicalize_products}'
    except:
        return ''

if __name__ == '__main__':
    total_syn_yield_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/reaxys_total_syn_yield_data.csv', encoding='utf-8')
    single_step_total_syn_reaction_df = total_syn_yield_df.loc[total_syn_yield_df['Reaction Details: Reaction Classification'] == 'Preparation']
    single_step_total_syn_reaction_df = single_step_total_syn_reaction_df.loc[~pd.isna(single_step_total_syn_reaction_df['Reaction'])]
    single_step_total_syn_reaction_df = single_step_total_syn_reaction_df.loc[~pd.isna(single_step_total_syn_reaction_df['Yield (numerical)'])]
    
    can_rxn_smiles_ls = []
    for i, rxn_smiles in tqdm(enumerate(single_step_total_syn_reaction_df['Reaction'].tolist()), total=len(single_step_total_syn_reaction_df)):
        can_rxn_smiles = canonicalize_reaction(rxn_smiles)
        can_rxn_smiles_ls.append(can_rxn_smiles)
    single_step_total_syn_reaction_df['can_rxn_smiles'] = can_rxn_smiles_ls
    single_step_total_syn_reaction_df = single_step_total_syn_reaction_df.loc[single_step_total_syn_reaction_df['can_rxn_smiles'] != '']
    
    rxn_mapper = RXNMapper()
    mapped_rxn_smiles = []
    confidence = []
    for rxn in tqdm(single_step_total_syn_reaction_df['can_rxn_smiles'].tolist()):
        try:
            results = rxn_mapper.get_attention_guided_atom_maps([rxn])[0]
            mapped_rxn_smiles.append(results['mapped_rxn'])
            confidence.append(results['confidence'])
        except:
            mapped_rxn_smiles.append('')
            confidence.append('')   
    single_step_total_syn_reaction_df['mapped_rxn_smiles'] = mapped_rxn_smiles
    single_step_total_syn_reaction_df['confidence'] = confidence
    single_step_total_syn_reaction_df = single_step_total_syn_reaction_df.loc[single_step_total_syn_reaction_df['mapped_rxn_smiles'] != '']
    total_syn_df = pd.DataFrame(single_step_total_syn_reaction_df, columns=['Reaction ID', 'Reaction', 'can_rxn_smiles', 'Yield (numerical)', 'Reaction Type',  'Time (Reaction Details) [h]',
                                                                   'Temperature (Reaction Details) [C]', 'Reagent', 'Catalyst', 'Solvent (Reaction Details)', 'Links to Reaxys', 'mapped_rxn_smiles'])

    not_equal_idx = []
    multi_data_idx = []
    delete_index = []
    for idx, (reaction, Yield) in tqdm(enumerate(zip(total_syn_df['Reaction'].tolist(), total_syn_df['Yield (numerical)'].tolist())), total=len(total_syn_df)):
        if (not pd.isna(reaction)) and (not pd.isna(Yield)):
            reaction_list = reaction.split('>>')
            product_list = reaction_list[1].split('.')
            Yield_str = str(Yield)
            Yield_list = Yield_str.split(';')
            if len(product_list) != len(Yield_list):
                not_equal_idx.append(idx)
                delete_index.append(idx)
            else:
                if len(product_list) != 1:
                    multi_data_idx.append(idx)
                    delete_index.append(idx)
    multi_data_df = total_syn_df.iloc[multi_data_idx]
    total_syn_yield_single_pdt_df = total_syn_df.drop(total_syn_df.index[delete_index]).reset_index(inplace=True)

    new_time_list = []
    new_temperature_list = []
    new_reagent_list = []
    new_catalyst_list = []
    new_solvent_list = []
    for idx, time in enumerate(total_syn_yield_single_pdt_df['Time (Reaction Details) [h]'].tolist()):
        if not pd.isna(time):
            time = time.split('; ')
            time = list(set(time))
            time.sort()
            new_time = '; '.join(time)
            new_time_list.append(new_time)
        else:
            new_time_list.append('')
    print(len(new_time_list))
    
    for idx, temperature in enumerate(total_syn_yield_single_pdt_df['Temperature (Reaction Details) [C]'].tolist()):
        if not pd.isna(temperature):
            temperature = temperature.split('; ')
            temperature = list(set(temperature))
            temperature.sort()
            new_temperature = '; '.join(temperature)
            new_temperature_list.append(new_temperature)
        else:
            new_temperature_list.append('')
    print(len(new_temperature_list))    
    
    for idx, reagent in enumerate(total_syn_yield_single_pdt_df['Reagent'].tolist()):
        if not pd.isna(reagent):
            reagent = reagent.split('; ')
            reagent = list(set(reagent))
            reagent.sort()
            new_reagent = '; '.join(reagent)
            new_reagent_list.append(new_reagent)
        else:
            new_reagent_list.append('')
    print(len(new_reagent_list))
    
    for idx, catalyst in enumerate(total_syn_yield_single_pdt_df['Catalyst'].tolist()):
        if not pd.isna(catalyst):
            catalyst = catalyst.split('; ')
            catalyst = list(set(catalyst))
            catalyst.sort()
            new_catalyst = '; '.join(catalyst)
            new_catalyst_list.append(new_catalyst)
        else:
            new_catalyst_list.append('')
    print(len(new_catalyst_list))
    
    for idx, solvent in enumerate(total_syn_yield_single_pdt_df['Solvent (Reaction Details)'].tolist()):
        if not pd.isna(solvent):
            solvent = solvent.split('; ')
            solvent = list(set(solvent))
            solvent.sort()
            new_solvent = '; '.join(solvent)
            new_solvent_list.append(new_solvent)
        else:
            new_solvent_list.append('')
    print(len(new_solvent_list))
    
    total_syn_yield_single_pdt_df['new_time'] = new_time_list
    total_syn_yield_single_pdt_df['new_temperature'] = new_temperature_list
    total_syn_yield_single_pdt_df['new_reagent'] = new_reagent_list
    total_syn_yield_single_pdt_df['new_catalyst'] = new_catalyst_list
    total_syn_yield_single_pdt_df['new_solvent'] = new_solvent_list
    
    iterrows_dict = defaultdict(list)
    need_row_name = ['Reaction ID', 'Reaction', 'can_rxn_smiles', 'Reaction Type', 'new_time', 'new_temperature', 
                     'new_reagent', 'new_catalyst', 'new_solvent', 'mapped_rxn_smiles', 'Links to Reaxys']
    for idx, row in tqdm(total_syn_yield_single_pdt_df.iterrows(), total=len(total_syn_yield_single_pdt_df)):
        row[pd.isna(row)] = ''
        key = '[SPLIT]'.join('%s' % i for i in(row[need_row_name].tolist()))
        iterrows_dict[key].append(row['Yield (numerical)'])
    iterrows_list = list(iterrows_dict.items())
    deduplicate_dict = {
        'Reaction ID': [],
        'Reaction': [],
        'can_rxn_smiles': [],
        'Reaction Type': [],
        'new_time': [],
        'new_temperature': [],
        'new_reagent': [],
        'new_catalyst': [],
        'new_solvent': [],
        'mapped_rxn_smiles': [],
        'Links to Reaxys': [],
        'Yield (numerical)': []
    }
    for key, value in iterrows_list:
        data = key.split('[SPLIT]')
        assert len(need_row_name) == len(data)
        for i in range(len(need_row_name)):
            deduplicate_dict[need_row_name[i]].append(data[i])
        deduplicate_dict['Yield (numerical)'].append(np.array(value).mean())
    total_syn_deduplicated_df = pd.DataFrame.from_dict(deduplicate_dict)
    total_syn_deduplicated_df.drop_duplicates(subset=['Reaction ID', 'Reaction', 'can_rxn_smiles', 'Reaction Type', 'new_time', 'new_temperature', 
                                           'new_reagent', 'new_catalyst', 'new_solvent', 'mapped_rxn_smiles', 'Yield (numerical)'], keep='first', inplace=True)
    rtn_id_ls = []
    dupli_rtn_idx_ls = []
    for rtn_id in tqdm(total_syn_deduplicated_df['Reaction ID'].tolist()):
        if rtn_id in rtn_id_ls: 
            dupli_rtn_idx_ls.append(rtn_id)
        else:
            rtn_id_ls.append(rtn_id)
    
    multi_yeild_idx_ls = []
    multi_reaction_id = []
    for idx, reaction_id in tqdm(enumerate(total_syn_deduplicated_df['Reaction ID'].tolist()), total=len(total_syn_deduplicated_df)):
        if reaction_id in dupli_rtn_idx_ls:
            multi_yeild_idx_ls.append(idx)
            multi_reaction_id.append(reaction_id)
        else:
            continue
    total_syn_multi_yield_df = total_syn_deduplicated_df.iloc[multi_yeild_idx_ls]
    print(total_syn_multi_yield_df.shape)

    rtn_yield_dict = defaultdict(list)
    for rtn, rtn_yield in tqdm(zip(total_syn_multi_yield_df['Reaction ID'].tolist(), total_syn_multi_yield_df['Yield (numerical)'].tolist()), total=len(total_syn_multi_yield_df)):
        rtn_yield_dict[rtn].append(rtn_yield)
    yield_difference_ls = []
    for rtn in tqdm(total_syn_multi_yield_df['Reaction ID'].tolist(), total=len(total_syn_multi_yield_df)):
        difference_value = max(rtn_yield_dict[rtn]) - min(rtn_yield_dict[rtn])
        yield_difference_ls.append(difference_value)
    total_syn_multi_yield_df['yield_difference'] = yield_difference_ls
    total_syn_multi_yield_df = total_syn_multi_yield_df.loc[total_syn_multi_yield_df['yield_difference'] >= 20] 
    total_syn_multi_yield_df.to_csv(os.path.join('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/total_syn_multi_yield_difference_20.csv'), encoding='utf-8', index=False)
            