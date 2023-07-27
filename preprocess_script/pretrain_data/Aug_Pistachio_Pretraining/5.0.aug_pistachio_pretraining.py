import pandas as pd


def selcet_n_sample(dataframe, N=5, select_N=2):
    dataframe_len = len(dataframe)
    one_select_index = [True] * select_N + [False] * (N-select_N)
    select_index = pd.Series(one_select_index * int(dataframe_len/N))
    select_dataframe = dataframe.loc[select_index]
    return select_dataframe

if __name__ == "__main__":
    all_trn_df = pd.read_csv('../../dataset/pretrain_data/aug_pistachio_pretraining_train.csv', encoding='utf-8')
    all_eval_df = pd.read_csv('../../dataset/pretrain_data/aug_pistachio_pretraining_eval.csv', encoding='utf-8')
    select_trn_df = selcet_n_sample(all_trn_df)
    select_eval_df = selcet_n_sample(all_eval_df)
    select_trn_df.to_csv('../../dataset/pretrain_data/aug_pistachio_pretraining_train_dataset.csv', encoding='utf-8', index=False)
    select_eval_df.to_csv('../../dataset/pretrain_data/aug_pistachio_pretraining_eval_dataset.csv', encoding='utf-8', index=False)        
    print('Done!')


