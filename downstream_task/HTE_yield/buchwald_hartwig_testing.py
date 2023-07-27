import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../../preprocess_script/benchmark_data')
from Buchwald_Hartwig_data import generate_buchwald_hartwig_rxns
from rxnfp.models import SmilesClassificationModel 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_model_from_results_folder(name, split, epoch=30, results_folder='../../outputs/buchwald_hartwig', model_type='bert'):
    models_folder = os.path.join(results_folder, f"{name}_split_{str(split).replace('-','_')}")
    model_path = [os.path.join(models_folder, o) for o in os.listdir(models_folder) 
                        if os.path.isdir(os.path.join(models_folder, o)) and o.endswith(f'epoch-{epoch}')][0]
    model = SmilesClassificationModel(model_type, 
                                      model_path, 
                                      num_labels=1, 
                                      args={
                                        "regression": True,
                                        "use_multiprocessing_for_evaluation":False,
                                        "use_multiprocessing":False
                                      }, 
                                      use_cuda=True if not debug else False, 
                                      cuda_device=0) 
    return model

def load_model_from_results_folder_test1_4(name, split, seed, epoch=30, results_folder='../../outputs/buchwald_hartwig', model_type='bert'):
    models_folder = os.path.join(results_folder, f"{name}_split_{str(split).replace('-','_')}_{seed}")
    model_path = [os.path.join(models_folder, o) for o in os.listdir(models_folder) 
                        if os.path.isdir(os.path.join(models_folder,o)) and o.endswith(f'epoch-{epoch}')][0]
    model = SmilesClassificationModel(model_type, 
                                      model_path, 
                                      num_labels=1, 
                                      args={"regression": True,
                                            "use_multiprocessing_for_evaluation":False,
                                            "use_multiprocessing":False}, 
                                            use_cuda=True if not debug else False, 
                                            cuda_device=0) 
    return model


if __name__ == "__main__":
    debug = True

    NAME_SPLIT = [
    ('FullCV_01', 2768),
    ('FullCV_02', 2768),
    ('FullCV_03', 2768),
    ('FullCV_04', 2768),
    ('FullCV_05', 2768),
    ('FullCV_06', 2768),
    ('FullCV_07', 2768),
    ('FullCV_08', 2768),
    ('FullCV_09', 2768),
    ('FullCV_10', 2768),
    ]
    OUT_OF_SAMPLE_SPLIT = [
    ('Test1', 3058),
    ('Test2', 3056),
    ('Test3', 3059),
    ('Test4', 3056),
    ]   
    DISCOVERY_SPLIT = [
    ('FullCV_01', 99), ('FullCV_01', 198), ('FullCV_01', 396), ('FullCV_01', 792), ('FullCV_01', 1187), ('FullCV_01', 1978),
    ('FullCV_02', 99), ('FullCV_02', 198), ('FullCV_02', 396), ('FullCV_02', 792), ('FullCV_02', 1187), ('FullCV_02', 1978),
    ('FullCV_03', 99), ('FullCV_03', 198), ('FullCV_03', 396), ('FullCV_03', 792), ('FullCV_03', 1187), ('FullCV_03', 1978),
    ('FullCV_04', 99), ('FullCV_04', 198), ('FullCV_04', 396), ('FullCV_04', 792), ('FullCV_04', 1187), ('FullCV_04', 1978),
    ('FullCV_05', 99), ('FullCV_05', 198), ('FullCV_05', 396), ('FullCV_05', 792), ('FullCV_05', 1187), ('FullCV_05', 1978),
    ('FullCV_06', 99), ('FullCV_06', 198), ('FullCV_06', 396), ('FullCV_06', 792), ('FullCV_06', 1187), ('FullCV_06', 1978), 
    ('FullCV_07', 99), ('FullCV_07', 198), ('FullCV_07', 396), ('FullCV_07', 792), ('FullCV_07', 1187), ('FullCV_07', 1978),
    ('FullCV_08', 99), ('FullCV_08', 198), ('FullCV_08', 396), ('FullCV_08', 792), ('FullCV_08', 1187), ('FullCV_08', 1978),
    ('FullCV_09', 99), ('FullCV_09', 198), ('FullCV_09', 396), ('FullCV_09', 792), ('FullCV_09', 1187), ('FullCV_09', 1978),
    ('FullCV_10', 99), ('FullCV_10', 198), ('FullCV_10', 396), ('FullCV_10', 792), ('FullCV_10', 1187), ('FullCV_10', 1978),
    ]

    r2_scores = []
    for (name, split) in NAME_SPLIT:
    # for (name, split) in DISCOVERY_SPLIT:
        df_doyle = pd.read_excel('../../dataset/source_dataset/Buchwald_Hartwig_reaction/Dreher_and_Doyle_input_data.xlsx', sheet_name=name)
        df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle)
        train_df = df_doyle.iloc[:split-1][['rxn', 'Output']] 
        test_df = df_doyle.iloc[split-1:][['rxn', 'Output']] 
        train_df.columns = ['text', 'labels']
        test_df.columns = ['text', 'labels']
        y_test = test_df.labels.values  
        mean = train_df.labels.mean()
        std = train_df.labels.std()    

        model = load_model_from_results_folder(name, split)
        y_preds = model.predict(test_df.text.values.tolist())[0]
        y_preds = y_preds * std + mean 
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

    for (name, split) in OUT_OF_SAMPLE_SPLIT:
        for seed in [42, 69, 2222, 2626]:
            df_doyle = pd.read_excel('../../dataset/source_dataset/Buchwald_Hartwig_reaction/Dreher_and_Doyle_input_data.xlsx', sheet_name=name)
            df_doyle['rxn'] = generate_buchwald_hartwig_rxns(df_doyle)
            train_df = df_doyle.iloc[:split-1][['rxn', 'Output']] 
            test_df = df_doyle.iloc[split-1:][['rxn', 'Output']] 
            train_df.columns = ['text', 'labels']
            test_df.columns = ['text', 'labels']
            y_test = test_df.labels.values 
            mean = train_df.labels.mean()
            std = train_df.labels.std()
            model = load_model_from_results_folder_test1_4(name, split, seed)
            y_preds = model.predict(test_df.text.values.tolist())[0]
            y_preds = y_preds * std + mean 
            y_preds = np.clip(y_preds, 0, 100)   
            r_squared = r2_score(y_test, y_preds)
            rmse = mean_squared_error(y_test, y_preds) ** 0.5
            mae = mean_absolute_error(y_test, y_preds)
            print(f"{name} | {seed} | R2 {r_squared:.4f} | RMSE {rmse:.4f} | MAE {mae:.4f}")

