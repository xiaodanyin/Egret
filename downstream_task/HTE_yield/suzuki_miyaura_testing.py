import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from rxnfp.models import SmilesClassificationModel
import os

def load_model_from_results_folder(name, split, epoch=160, results_folder='../../outputs/suzuki_miyaura', model_type='bert'):
    models_folder = os.path.join(results_folder, f"{name}_split_{str(split).replace('-','_')}")
    model_path = [os.path.join(models_folder, o) for o in os.listdir(models_folder) 
                        if os.path.isdir(os.path.join(models_folder,o)) and o.endswith(f'epoch-{epoch}')][0]
    print(model_path)
    model = SmilesClassificationModel(model_type, 
                                      model_path,
                                      num_labels=1, 
                                      args={
                                      "regression": True,
                                      "use_multiprocessing_for_evaluation":False,
                                      "use_multiprocessing":False
                                      }, 
                                      use_cuda=False, 
                                      cuda_device=1)  
    return model

if __name__ == "__main__":

    NAME_SPLIT = [
    ('random_split_0', 4032),
    ('random_split_1', 4032),
    ('random_split_2', 4032),
    ('random_split_3', 4032),
    ('random_split_4', 4032),
    ('random_split_5', 4032),
    ('random_split_6', 4032),
    ('random_split_7', 4032),
    ('random_split_8', 4032),
    ('random_split_9', 4032),
    ]
    
    r2_scores = []
    for (name, split) in NAME_SPLIT:
        df = pd.read_csv(f'../../dataset/source_dataset/Suzuki_Miyaura_reaction/random_splits/{name}.tsv', sep='\t')
        train_df = df.iloc[:split][['rxn', 'y']]
        test_df = df.iloc[split:][['rxn', 'y']]
        train_df.columns = ['text', 'labels']
        test_df.columns = ['text', 'labels']
        y_test = test_df['labels'].values * 100 
        mean = train_df.labels.mean()
        std = train_df.labels.std()
        model = load_model_from_results_folder(name, split)
        y_preds = model.predict(test_df.text.values.tolist())[0]
        y_preds = y_preds * std + mean 
        y_preds = y_preds * 100
        y_preds = np.clip(y_preds, 0, 100)   
        r_squared = r2_score(y_test, y_preds)
        r2_scores.append(r_squared)
        rmse = mean_squared_error(y_test, y_preds) ** 0.5
        mae = mean_absolute_error(y_test, y_preds)
        print(f"{name} | R2 {r_squared:.2f} | RMSE {rmse:.1f} | MAE {mae:.1f}")
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    print(str(r2_mean))
    print(str(r2_std))