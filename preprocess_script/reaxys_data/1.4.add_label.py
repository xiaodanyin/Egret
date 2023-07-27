import pandas as pd


if __name__ == "__main__":
    split_ls = ['trn', 'val', 'test']
    for split in split_ls:    
        reaxys_yield_data_df = pd.read_csv(f'../../dataset/source_dataset/Reaxys-MultiCondi-Yield/reaxys_{split}_dataset_pred_category.csv', encoding='utf-8')
        yield_ls = reaxys_yield_data_df['Yield (numerical)'].tolist()
        yield_label_ls = []
        for true_yield in yield_ls:
            if 80 <= true_yield <= 100:
                yield_label_ls.append('0')
            elif 50 <= true_yield < 80:
                yield_label_ls.append('1')
            elif 30 <= true_yield < 50:
                yield_label_ls.append('2')
            elif 0 <= true_yield < 30:
                yield_label_ls.append('3')
        assert len(yield_ls) == len(yield_label_ls)
        reaxys_yield_data_df['yield_label'] = yield_label_ls
        reaxys_yield_data_df.to_csv(f'../../dataset/source_dataset/Reaxys-MultiCondi-Yield/labeled_category_{split}_dataset.csv', index=False)