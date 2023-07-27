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
    test_df = pd.read_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/labeled_category_test_dataset.csv', encoding='utf-8')

    trn_df['rxn_main_category'] = main_category(trn_df)
    trn_df = trn_df.loc[trn_df['rxn_main_category'].isin([0])]      # [0, 6, 9, 10, 11]   
    val_df['rxn_main_category'] = main_category(val_df)
    val_df = val_df.loc[val_df['rxn_main_category'].isin([0])]      # [0, 6, 9, 10, 11]   
    test_df['rxn_main_category'] = main_category(test_df)
    test_df = test_df.loc[test_df['rxn_main_category'].isin([0])]      # [0, 6, 9, 10, 11]   

    meta_testing_support_set = []
    for smi, label, rxn_category in zip(trn_df['fin_can_rxn_smi'].tolist(), trn_df['yield_lable'].tolist(), trn_df['rxn_main_category'].tolist()):
        task_dict = {'text': smi, 'label': label, 'domain': rxn_category}
        meta_testing_support_set.append(task_dict)
    meta_testing_query_set = []
    for smi, label, rxn_category in zip(val_df['fin_can_rxn_smi'].tolist(),  val_df['yield_lable'].tolist(), val_df['rxn_main_category'].tolist()):
        task_dict = {'text': smi, 'label': label, 'domain': rxn_category}
        meta_testing_query_set.append(task_dict)
    meta_testing_spt_json_file_path = '../../dataset/source_dataset/Reaxys-MultiCondi-Yield/meta_testing_support_set_category_0.json'
    meta_testing_spt_json_file = open(meta_testing_spt_json_file_path, mode='w')
    json.dump(meta_testing_support_set, meta_testing_spt_json_file)
    meta_testing_qry_json_file_path = '../../dataset/source_dataset/Reaxys-MultiCondi-Yield/meta_testing_query_set_category_0.json'
    meta_testing_qry_json_file = open(meta_testing_qry_json_file_path, mode='w')
    json.dump(meta_testing_query_set, meta_testing_qry_json_file)
    test_df.to_csv('../../dataset/source_dataset/Reaxys-MultiCondi-Yield/test_dataset_category_0.csv')