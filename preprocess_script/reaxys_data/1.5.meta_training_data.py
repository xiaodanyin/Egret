import pandas as pd
import json

def main_category(df):
    rxn_main_category = []
    for category in df['rxn_category'].tolist():
        main_category = eval(str(category).split('.')[0])
        rxn_main_category.append(main_category)
    assert len(rxn_main_category) == len(df)    
    return rxn_main_category



if __name__ == "__main__":
    trn_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/labeled_category_trn_dataset.csv', encoding='utf-8')
    val_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/labeled_category_val_dataset.csv', encoding='utf-8')

    trn_df['rxn_main_category'] = main_category(trn_df)
    trn_df = trn_df.loc[trn_df['rxn_main_category'].isin([1, 2, 3, 4, 5, 7, 8])]
    val_df['rxn_main_category'] = main_category(val_df)
    val_df = val_df.loc[val_df['rxn_main_category'].isin([1, 2, 3, 4, 5, 7, 8])]

    meta_training_task = []
    for smi, label, rxn_category in zip(trn_df['fin_can_rxn_smi'].tolist(), trn_df['yield_label'].tolist(), trn_df['rxn_main_category'].tolist()):
        task_dict = {'text': smi, 'label': label, 'domain': rxn_category}
        meta_training_task.append(task_dict)
    for smi, label, rxn_category in zip(val_df['fin_can_rxn_smi'].tolist(), val_df['yield_label'].tolist(), val_df['rxn_main_category'].tolist()):
        task_dict = {'text': smi, 'label': label, 'domain': rxn_category}
        meta_training_task.append(task_dict)
    print(len(meta_training_task)) 
    json_file_path = '../../dataset/source_dataset/Reaxys-MultiCondi-Yield/meta_training_task_dataset.json'
    json_file = open(json_file_path, mode='w')
    json.dump(meta_training_task, json_file)