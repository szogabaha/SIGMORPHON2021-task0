import hydra
import pandas as pd
from omegaconf import DictConfig
import numpy as np
import os
import csv


@hydra.main(config_name="evaluate_config")
def main(cfg: DictConfig):
    src_file = os.path.join(cfg.target_dir, f'{cfg.lang}.{cfg.extension}')
    df = pd.read_csv(src_file, sep="\t", header=None, names=['lemma', 'infl', 'tags'])
    df = df.drop_duplicates()
    df = df.replace(np.nan, 'nan', regex=True)
    target = df.infl.to_numpy()
    

    pred_file = os.path.join(cfg.pred_dir, f'{cfg.lang}_predicted.txt')
    with open(pred_file, 'r') as file:
        pred = np.array(file.read().split('\n'))
    correct = (target == pred).sum()
    acc = correct / len(pred)

    result_file = os.path.join(cfg.pred_dir, 'result_csv')

    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cfg.lang, acc])


if __name__ == '__main__':
    main()
