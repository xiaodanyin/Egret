import pandas as pd
import os

if __name__ == '__main__':
    uspto_data = ['gram', 'milligram']
    for uspto in uspto_data:
        train_df = pd.read_csv(os.path.join(f'../../dataset/source_dataset/USPTO_yield/{uspto}_train_random_split.tsv'), sep='\t', index_col=0)
        train_df.columns = ['text', 'labels']

        val_df = train_df.sample(frac=0.1, random_state=147)
        trn_df = train_df.drop(val_df.index).reset_index(drop=True)  
        val_df = val_df.reset_index(drop=True)

        val_df.to_csv(f'../../dataset/source_dataset/USPTO_yield/{uspto}_val_random_split.csv', encoding='utf-8', index=False)
        trn_df.to_csv(f'../../dataset/source_dataset/USPTO_yield/{uspto}_trn_random_split.csv', encoding='utf-8', index=False)               