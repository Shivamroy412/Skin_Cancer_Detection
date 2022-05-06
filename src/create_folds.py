import os
import pandas as pd
from sklearn import model_selection
import config

if __name__ == "__main__":

    if config.training_local:
        input_path = os.path.join(config.LOCAL_ROOT_PATH, config.INPUT_PATH[1:])
        data_path = os.path.join(config.LOCAL_ROOT_PATH, config.DATA_PATH[1:])
        #Sliced the path above to remove the leading '/' seperator


    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop = True)
    
    y = df.target.values

    kfold_model = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 49 )

    for fold_, (train_idx_, valid_idx_) in enumerate(kfold_model.split(X=df, y=y)):
        df.loc[valid_idx_ , "kfold"] = fold_

    
    os.makedirs(input_path, exist_ok = True)
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index = False)
    
    print(df.kfold.value_counts())