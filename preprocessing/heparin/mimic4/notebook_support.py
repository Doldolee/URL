"""Pure-array support for the copied, corrected Heparin-IV notebook."""
import numpy as np


def preserved_split_trajectories(trajectory_ids, stay_ids, split_manifest):
    """Reuse recorded ICU-stay membership instead of resplitting by new reward."""
    pairs = set(zip(np.asarray(trajectory_ids, dtype=np.int64), np.asarray(stay_ids, dtype=np.int64)))
    by_stay = {}
    seen_trajectories = set()
    for trajectory, stay in pairs:
        if stay in by_stay or trajectory in seen_trajectories:
            raise ValueError("Expected exactly one trajectory per ICU stay")
        by_stay[int(stay)] = int(trajectory)
        seen_trajectories.add(int(trajectory))
    stored = split_manifest['icustayids']
    all_stays = [int(s) for split in ['train', 'val', 'test'] for s in stored[split]]
    if len(set(all_stays)) != len(all_stays):
        raise ValueError("Recorded splits overlap")
    if set(all_stays) != set(by_stay):
        raise ValueError("Cohort differs from the saved split manifest; do not silently resplit")
    return {split: np.asarray(sorted(by_stay[int(s)] for s in stored[split]), dtype=np.int64)
            for split in ['train', 'val', 'test']}
