"""Load MIMIC-III Heparin arrays and recover complete patient groups."""
from pathlib import Path
import numpy as np
from util import FIELDS
from util import recover_heparin_subject_groups


def recover(root, demog):
    root = Path(root)
    arrays = {split: {name: np.load(root/'dataset/heparin'/f'{split}_{name}.npy')
                      for name in FIELDS} for split in ['train', 'test']}
    for data in arrays.values():
        data['done'] = 1 - data['done']
    cohort = root/'preprocessing/heparin/mimic3/final_result/1hourly_discrete/no_reward_SAH/heparin_final_data_withTimes.csv'
    groups, stays, keep, report = recover_heparin_subject_groups(arrays, cohort, demog)
    return arrays, groups, stays, keep, report
