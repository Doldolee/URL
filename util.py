"""Replay buffers, clinical data helpers, patient splits, paths and artifact loading."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import numpy as np
import random
import sys
import threading
import types
import time
import torch
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


# Project paths

def project_root(override=None):
    return Path(override).expanduser().resolve() if override is not None else Path(__file__).resolve().parent


def project_path(path, root=None):
    path = Path(path)
    return path if path.is_absolute() else project_root(root) / path


# Artifact IO and source provenance

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


SOURCE_PACKAGES = ("scripts", "configs", "tests")


def active_source_paths(root):
    root = Path(root)
    candidates = list(root.glob("*.py"))
    candidates += [path for folder in SOURCE_PACKAGES for path in (root / folder).rglob("*.py")]
    return sorted(path for path in candidates if not path.name.startswith("._") and "__pycache__" not in path.parts)


def source_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256_file(path) for path in active_source_paths(root)}


def verify_source_manifest(root):
    """Check the post-migration runtime instead of demanding archived filenames."""
    root = Path(root)
    manifest = json.loads((root / "source_manifest.json").read_text())
    actual = source_hashes(root)
    if actual != manifest["sources"]:
        changed = sorted(set(actual) ^ set(manifest["sources"]) |
                         {name for name in actual.keys() & manifest["sources"].keys()
                          if actual[name] != manifest["sources"][name]})
        raise ValueError("Current source manifest mismatch: " + ", ".join(changed))
    for relative, expected in manifest.get("interfaces", {}).items():
        if sha256_file(root / relative) != expected:
            raise ValueError("Current interface manifest mismatch: " + relative)
    return manifest


# Dataset loading and validation

NUM_ACTIONS = 25


def validate_data(state, action, done, num_actions=NUM_ACTIONS):
    state = np.asarray(state)
    action = np.asarray(action).reshape(-1)
    if state.ndim != 2 or len(state) != len(action) or len(action) != np.asarray(done).size:
        raise ValueError("state/action/done dimensions disagree")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("non-finite states or actions")
    if not np.array_equal(action, action.astype(np.int64)):
        raise ValueError("actions must be integers")
    action = action.astype(np.int64)
    if not ((action >= 0) & (action < num_actions)).all():
        raise ValueError(f"actions must be in [0, {num_actions-1}]")
    episode_slices(done)
    return state, action


def load_split(dataset_dir, split):
    data = {}
    paths = {}
    for field in ["state", "action", "next_state", "reward", "done"]:
        path = Path(dataset_dir) / (split + "_" + field + ".npy")
        data[field] = np.load(path, mmap_mode="r", allow_pickle=False)
        paths[field] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                        "shape": list(data[field].shape), "dtype": str(data[field].dtype)}
    validate_data(data["state"], data["action"], data["done"])
    if data["next_state"].shape != data["state"].shape:
        raise ValueError("next_state shape disagrees")
    for field in ["reward", "done"]:
        if data[field].size != len(data["state"]):
            raise ValueError(field + " length disagrees")
    return data, paths


FIELDS = ['state','next_state','action','reward','done']


REWARD_TYPES = {'sepsis': 'terminal', 'heparin': 'per_step'}


def reward_type(dataset):
    if dataset not in REWARD_TYPES:
        raise ValueError("Patient-verified training currently supports 'heparin' and 'sepsis' (MIMIC-III)")
    return REWARD_TYPES[dataset]


def buffer_arrays(buffer):
    return {name: getattr(buffer, name)[:buffer.crt_size] for name in FIELDS}


def prepare_subject_split(dataset, train_buffer, test_buffer, project_root, *, cohort_csv=None, demog_csv=None):
    """Remove test patients before constructing or fitting the evaluated policy."""
    reward_type(dataset)
    root = Path(project_root)
    if dataset == 'heparin':
        cohort = cohort_csv or root/'preprocessing/heparin/mimic3/final_result/1hourly_discrete/no_reward_SAH/heparin_final_data_withTimes.csv'
        demog = demog_csv or root.parent/'heparin_RL/preprocessing/mimic3/demog.csv'
        groups, _, keep, report = recover_heparin_subject_groups(
            {'train': buffer_arrays(train_buffer), 'test': buffer_arrays(test_buffer)}, cohort, demog)
        train_groups, test_groups = groups['train'], groups['test']
    else:
        local_cohort = root/'preprocessing/sepsis/mimic3/sepsis_final_data_withTimes.csv'
        cohort = cohort_csv or (local_cohort if local_cohort.is_file() else DEFAULT_COHORT)
        train_groups, test_groups, keep, report = recover_subject_groups(
            buffer_arrays(train_buffer), buffer_arrays(test_buffer), cohort, demog_csv or DEFAULT_DEMOG)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (train_buffer.crt_size,) or not keep.any():
        raise ValueError('Invalid patient exclusion mask')
    for rows in episode_slices(train_buffer.done):
        if not np.all(keep[rows] == keep[rows.start]):
            raise ValueError('Patient exclusion must retain or remove whole episodes')
    if np.intersect1d(np.asarray(train_groups)[keep], test_groups).size:
        raise ValueError('Test subjects remain after patient exclusion')
    for name in FIELDS + ['bc_prob']:
        if hasattr(train_buffer, name):
            setattr(train_buffer, name, getattr(train_buffer, name)[:train_buffer.crt_size][keep].copy())
    train_buffer.crt_size = int(keep.sum())
    return np.asarray(train_groups)[keep], np.asarray(test_groups), report


# Complete trajectories and patient splits

def patient_split(groups, test_groups, reward, done, fraction=.2, seed=20261007):
    """Preserve the old test cohort and remove all its patients before splitting."""
    groups = np.asarray(groups).reshape(-1)
    done = np.asarray(done).reshape(-1)
    reward = np.asarray(reward).reshape(-1)
    if not (len(groups) == len(done) == len(reward)):
        raise ValueError('Groups, rewards, and done must align')
    for rows in episode_slices(done):
        if np.unique(groups[rows]).size != 1:
            raise ValueError('Each complete episode must belong to one patient')
    keep = ~np.isin(groups, np.unique(test_groups))
    kept = np.flatnonzero(keep)
    train_local, val_local, _, _, _ = _trajectory_split(
        done[keep], reward[keep], fraction, seed, groups[keep])
    train, val = kept[train_local], kept[val_local]
    assert not np.intersect1d(groups[train], groups[val]).size
    assert not np.intersect1d(groups[keep], test_groups).size
    return train, val, np.flatnonzero(~keep)


def episode_slices(done):
    """Return every complete trajectory, including the final trajectory."""
    done = np.asarray(done).reshape(-1)
    if not len(done) or not np.isin(done, [0, 1]).all() or done[-1] != 1:
        raise ValueError("done must be binary, nonempty, and end in a terminal row")
    stops = np.flatnonzero(done == 1) + 1
    starts = np.r_[0, stops[:-1]]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops)]


def episode_groups(done, offset=0):
    groups = np.empty(np.asarray(done).size, dtype=np.int64)
    for i, rows in enumerate(episode_slices(done)):
        groups[rows] = i + offset
    return groups


def trajectory_fingerprint(state, next_state, action, reward, done):
    """Canonical full-transition fingerprint; matches float32 tuple export."""
    digest = hashlib.sha256()
    for name, values, dtype in [
        ("state", state, "<f4"), ("next_state", next_state, "<f4"),
        ("action", np.asarray(action).reshape(-1), "<i8"),
        ("reward", np.asarray(reward).reshape(-1), "<f4"),
        ("done", np.asarray(done).reshape(-1), "u1"),
    ]:
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def grouped_partition(done, groups=None, random_seed=42, calibration_fraction=.2, validation_fraction=.2):
    """A deterministic disjoint group split; fractions refer to group counts."""
    slices = episode_slices(done)
    if groups is None:
        groups = episode_groups(done)
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != np.asarray(done).size:
        raise ValueError("groups length disagrees")
    for rows in slices:
        if len(np.unique(groups[rows])) != 1:
            raise ValueError("each done-delimited episode must belong to one group")
    if not 0 < calibration_fraction < 1 or not 0 < validation_fraction < 1 or calibration_fraction + validation_fraction >= 1:
        raise ValueError("positive calibration and validation fractions must sum to less than one")
    unique = np.unique(groups)
    shuffled = np.random.RandomState(random_seed).permutation(unique)
    n_cal = int(math.ceil(len(unique) * calibration_fraction))
    n_val = int(math.ceil(len(unique) * validation_fraction))
    if len(unique) - n_cal - n_val < 1:
        raise ValueError("too few independent groups for a three-way split")
    group_sets = {"calibration": shuffled[:n_cal], "validation": shuffled[n_cal:n_cal+n_val],
                  "fit": shuffled[n_cal+n_val:]}
    return {name: np.flatnonzero(np.isin(groups, values)) for name, values in group_sets.items()}


def initial_state_indices(done) -> np.ndarray:
    """Start rows of complete, contiguous episodes; done=1 means terminal."""
    raw = np.asarray(done)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("done must be a nonempty vector.")
    if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("done must be binary, with 1 indicating termination.")
    if raw[-1] != 1:
        raise ValueError("The final row must terminate a complete episode.")
    ends = np.flatnonzero(raw == 1)
    return np.r_[0, ends[:-1] + 1].astype(np.int64)


def _trajectory_split(done: np.ndarray, reward: np.ndarray, fraction: float, seed: int, groups=None):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    ends = np.flatnonzero(done == 1)
    if ends.size < 2:
        raise ValueError("At least two training episodes are needed for holdout.")
    episode_for_row = np.r_[0, np.cumsum(done[:-1], dtype=np.int64)]
    starts = np.r_[0, ends[:-1] + 1]
    if groups is None:
        group_for_episode = np.arange(len(ends), dtype=np.int64)
        group_count = len(ends)
    else:
        supplied_groups = np.asarray(groups)
        if supplied_groups.shape == (len(done), 1):
            supplied_groups = supplied_groups[:, 0]
        if supplied_groups.shape == (len(done),):
            episode_groups = supplied_groups[starts]
            if not np.equal(supplied_groups, episode_groups[episode_for_row]).all():
                raise ValueError("Every trajectory must belong to exactly one split group.")
        elif supplied_groups.shape == (len(ends),):
            episode_groups = supplied_groups
        else:
            raise ValueError("groups must contain one label per row or per trajectory.")
        unique_groups, group_for_episode = np.unique(episode_groups, return_inverse=True)
        group_count = len(unique_groups)
    if group_count < 2:
        raise ValueError("At least two independent split groups are needed for holdout.")
    rng = np.random.default_rng(seed)
    val_groups = []
    # Group-level terminal-outcome sign; mixed-outcome patients remain intact.
    outcome_sign = np.sign(np.bincount(group_for_episode, weights=reward[ends], minlength=group_count))
    for outcome in [-1, 0, 1]:
        group_indices = np.flatnonzero(outcome_sign == outcome)
        if len(group_indices) < 2:
            continue
        number = max(1, min(len(group_indices) - 1, round(len(group_indices) * fraction)))
        val_groups.extend(rng.permutation(group_indices)[:number].tolist())
    if not val_groups:
        val_groups = rng.permutation(group_count)[:1].tolist()
    val_groups = np.sort(np.asarray(val_groups, dtype=np.int64))
    val_episodes = np.flatnonzero(np.isin(group_for_episode, val_groups))
    val_mask = np.isin(episode_for_row, val_episodes)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), val_episodes, group_count, val_groups


# Training-state preprocessing

class TrainScaler:
    """Mean and scale fitted on training states only, with no clipping."""
    def fit(self, state):
        x = np.asarray(state, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite states')
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1.
        return self

    def transform(self, state):
        result = ((np.asarray(state, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite transformed state')
        return np.ascontiguousarray(result)


# Runtime controls

@contextmanager
def _cpu_threads(num_threads: int):
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Replay buffers

class ArrayTrainingBuffer:
    """Same uniform transition sampling, with an explicit paired seed stream."""
    def __init__(self, arrays, *, seed, batch_size=64):
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.arrays = [torch.as_tensor(np.ascontiguousarray(arrays[k]), dtype=(torch.int64 if k == 'action' else torch.float32))
                       for k in ['state', 'action', 'next_state', 'reward', 'done']]
        self.size = len(self.arrays[0])

    def sample(self):
        indices = torch.from_numpy(self.rng.integers(0, self.size, self.batch_size))
        result = tuple(x[indices] for x in self.arrays)
        return result + (torch.zeros((self.batch_size, 1)),)


class ReplayBuffer:
    def __init__(self, state_dim, batch_size, target_data, buffer_path,
                 buffer_size=200000, device='cuda'):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.buffer_path = Path(buffer_path)
        self.target_data = target_data
        self.state_dim = state_dim
        self.ptr = self.crt_size = 0
        # MIMIC-III Heparin stores continuation flags; IV stores terminal flags.
        self.stored_terminal_value = 0 if target_data == 'heparin' else 1
        self.done_convention = 'terminal_is_one'

    def load_data(self, size=-1, only_test_set=False):
        split = 'test' if only_test_set else 'train'
        arrays = {name: np.load(self.buffer_path/f'{split}_{name}.npy', allow_pickle=False)
                  for name in ['state', 'next_state', 'action', 'reward', 'done']}
        total = len(arrays['reward'])
        if total < 1 or any(len(value) != total for value in arrays.values()):
            raise ValueError('Empty or misaligned transition arrays')
        count = min(total, self.max_size, int(size) if size > 0 else self.max_size)
        if arrays['state'].shape[1:] != (self.state_dim,) or arrays['next_state'].shape != arrays['state'].shape:
            raise ValueError('State dimensions disagree with the buffer configuration')
        if not np.isin(arrays['done'], [0, 1]).all():
            raise ValueError('Stored done flags must be binary')
        action = arrays['action']
        if not np.isfinite(action).all() or not np.equal(action, np.floor(action)).all():
            raise ValueError('Stored actions must be finite integers')
        for name, values in arrays.items():
            dtype = np.int64 if name == 'action' else np.float32
            setattr(self, name, np.asarray(values[:count], dtype=dtype).copy())
        if self.stored_terminal_value == 0:
            self.done = 1 - self.done
        if self.done.reshape(-1)[-1] != 1:
            raise ValueError('Buffer truncation leaves an unfinished trajectory')
        self.crt_size = count
        # The sixth agent sample field is unused. Scalar BC_prob files are
        # never loaded or used as evaluation propensities.
        self.bc_prob = np.zeros((count, 1), dtype=np.float32)
        print(f'{self.target_data} Replay Buffer loaded: {count} transitions; done=1 terminal.')
        return self

    def sample(self):
        indices = np.random.randint(0, self.crt_size, size=self.batch_size)
        return tuple(torch.as_tensor(getattr(self, name)[indices],
                                     dtype=torch.int64 if name == 'action' else torch.float32,
                                     device=self.device)
                     for name in ['state', 'action', 'next_state', 'reward', 'done', 'bc_prob'])


# Heparin clinical data helpers

FEATURES = {'o:gender','o:age','o:Weight_kg','o:GCS','o:SysBP','o:DiaBP',
            'o:RR','o:Temp_C','o:Hb','o:WBC_count','o:Platelets_count',
            'o:PT','o:Arterial_lactate','o:Creatinine','o:Total_bili','o:INR'}


def recover_heparin_subject_groups(arrays, cohort, demog):
    """Match full standardized transitions to source stays; reject ambiguous IDs."""
    cohort, demog = Path(cohort), Path(demog)
    targets, groups, stays = {}, {}, {}
    for split, data in arrays.items():
        groups[split] = np.zeros(len(data['state']), dtype=np.int64)
        stays[split] = np.zeros(len(data['state']), dtype=np.int64)
        for rows in episode_slices(data['done']):
            key = trajectory_fingerprint(*[data[name][rows] for name in FIELDS])
            targets.setdefault(key, []).append((split, rows))
    with demog.open(newline='') as stream:
        subjects = {}
        for row in csv.DictReader(stream):
            stay = int(float(row['icustay_id']))
            subject = int(float(row['subject_id']))
            if stay in subjects and subjects[stay] != subject:
                raise ValueError('Ambiguous demographic ICU-to-subject mapping')
            subjects[stay] = subject
    found = {}
    with cohort.open(newline='') as stream:
        reader = csv.DictReader(stream)
        features = [name for name in reader.fieldnames if name in FEATURES]
        if len(features) != 16:
            raise ValueError('Expected all 16 Heparin state features')

        def emit(rows):
            rows = sorted(rows, key=lambda row: int(row['step']))
            if [int(row['step']) for row in rows] != list(range(len(rows))):
                raise ValueError('Invalid source trajectory steps')
            stay = int(float(rows[0]['m:icustayid']))
            if any(int(float(row['m:icustayid'])) != stay for row in rows):
                raise ValueError('Trajectory crosses ICU stays')
            state = np.array([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
            next_state = np.vstack([state[1:], state[-1:]])
            action = np.array([float(row['a:action']) for row in rows])[:, None]
            reward = np.r_[[float(row['PTT_reward']) for row in rows[1:]], 0.].astype(np.float32)[:, None]
            done = np.zeros((len(rows), 1))
            done[-1] = 1
            key = trajectory_fingerprint(state, next_state, action, reward, done)
            if key in targets:
                if key in found and found[key] != stay:
                    raise ValueError('Ambiguous trajectory mapping')
                if stay not in subjects:
                    raise ValueError('Matched ICU stay has no patient identifier')
                found[key] = stay

        current, rows = None, []
        for row in reader:
            if current is not None and row['traj'] != current:
                emit(rows)
                rows = []
            current = row['traj']
            rows.append(row)
        if rows:
            emit(rows)
    if set(found) != set(targets):
        raise ValueError(f'{len(set(targets)-set(found))} episodes did not map')
    for key, entries in targets.items():
        for split, rows in entries:
            stay = found[key]
            groups[split][rows] = subjects[stay]
            stays[split][rows] = stay
    keep = ~np.isin(groups['train'], np.unique(groups['test']))
    if not keep.any():
        raise ValueError('Patient exclusion removed every training episode')
    report = {
        'method': 'all state/next_state/action/shifted_reward/done rows of each trajectory hashed to CSV and ICU-stay demog',
        'cohort_path': str(cohort), 'cohort_sha256': sha256_file(cohort),
        'demog_path': str(demog), 'demog_sha256': sha256_file(demog),
        'features': features, 'original_done_semantics': '0=terminal; converted in memory to 1=terminal',
        'matched_train_episodes': len(episode_slices(arrays['train']['done'])),
        'matched_test_episodes': len(episode_slices(arrays['test']['done'])),
        'overlapping_train_test_subjects': len(np.intersect1d(groups['train'], groups['test'])),
        'excluded_train_rows': int((~keep).sum()),
        'excluded_train_episodes': sum(not keep[rows.start] for rows in episode_slices(arrays['train']['done'])),
        'test_subjects': len(np.unique(groups['test'])), 'test_rows': len(groups['test']),
    }
    return groups, stays, keep, report


STATE_FEATURES = (
    "o:gender", "o:age", "o:Weight_kg", "o:GCS", "o:SysBP", "o:DiaBP",
    "o:RR", "o:Temp_C", "o:Hb", "o:WBC_count", "o:Platelets_count", "o:PT",
    "o:Arterial_lactate", "o:Creatinine", "o:Total_bili", "o:INR",
)


def measured_aptt_reward(aptt):
    """III's smooth therapeutic-window reward; missing measurement gives 0.

    2 sigmoid(x-60) - 2 sigmoid(x-100) - 1, evaluated without overflow.
    NaN denotes absence of a measurement, not an aPTT of zero.
    """
    x = np.asarray(aptt, dtype=np.float64)
    if np.any(np.isinf(x)):
        raise ValueError("Infinite aPTT measurement")
    out = np.zeros(x.shape, dtype=np.float64)
    good = ~np.isnan(x)
    out[good] = np.tanh((x[good] - 60) / 2) - np.tanh((x[good] - 100) / 2) - 1
    return out


def measured_bin_means(events, stays, starts, width=3600):
    """Mean finite original measurements in each existing inclusive time bin.

    events is a last-write-wins {(stay, charttime): aPTT} mapping, so carrying
    forward a measurement never increases its weight or creates new rewards.
    """
    if len(stays) != len(starts) or width <= 0:
        raise ValueError("Invalid bin specification")
    grouped = defaultdict(list)
    for (stay, when), value in events.items():
        if np.isinf(value):
            raise ValueError("Infinite aPTT measurement")
        if np.isfinite(value):
            grouped[int(stay)].append((float(when), float(value)))
    ordered = {}
    for stay, entries in grouped.items():
        a = np.asarray(sorted(entries), dtype=np.float64)
        ordered[stay] = (a[:, 0], a[:, 1])
    means = np.full(len(stays), np.nan)
    counts = np.zeros(len(stays), dtype=np.int64)
    for i, (stay, start) in enumerate(zip(stays, starts)):
        if int(stay) not in ordered:
            continue
        times, values = ordered[int(stay)]
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, start + width, side="right")
        counts[i] = hi - lo
        if hi > lo:
            means[i] = np.mean(values[lo:hi])
    return means, counts


def transition_fingerprint(states, actions):
    """Match complete stored trajectories after their FloatTensor conversion."""
    s = np.asarray(states, dtype="<f4")
    a = np.asarray(actions, dtype="<f4").reshape(-1, 1)
    return hashlib.sha256(s.tobytes() + a.tobytes()).hexdigest()


def shift_rewards(row_rewards):
    """Stored buffer: r_t = CSV reward at next retained row; terminal r = 0."""
    values = np.asarray(row_rewards, dtype=np.float32).reshape(-1)
    if not len(values):
        raise ValueError("Empty trajectory")
    return np.r_[values[1:], np.float32(0)].astype(np.float64).reshape(-1, 1)


# Sepsis clinical data helpers

ROOT = project_root()


ORIGINAL = ROOT.parent / "multimodal_RL_sepsis/sepsis/sepsis_rl_code"


DEFAULT_COHORT = ORIGINAL / "dataset/mimic3/sepsis_final_data_withTimes.csv"


DEFAULT_DEMOG = ORIGINAL / "preprocess/mimic3/mimic-code-main/processed_files/demog.csv"


OBS_COLUMNS = {
    "o:GCS", "o:HR", "o:SysBP", "o:MeanBP", "o:DiaBP", "o:RR",
    "o:Temp_C", "o:FiO2_1", "o:Potassium", "o:Sodium", "o:Chloride",
    "o:Glucose", "o:Magnesium", "o:Calcium", "o:Ionised_Ca", "o:CO2_mEqL",
    "o:Hb", "o:WBC_count", "o:Platelets_count", "o:PTT", "o:PT",
    "o:Arterial_pH", "o:paO2", "o:paCO2", "o:HCO3", "o:Arterial_lactate",
    "o:SOFA", "o:SIRS", "o:Shock_Index", "o:PaO2_FiO2", "o:SpO2",
    "o:BUN", "o:Creatinine", "o:SGOT", "o:SGPT", "o:Total_bili",
    "o:Albumin", "o:INR", "o:output_4hourly",
}


DEM_COLUMNS = {"o:gender", "o:re_admission", "o:age", "o:Weight_kg"}


def recover_subject_groups(train, test, cohort_csv, demog_csv):
    """Identify current episodes by all five transition fields, then link IDs.

    No reliance on the old random split seed, CSV row order across trajectories,
    or approximate nearest-neighbor matching.  Partial/ambiguous matches fail.
    Original source identifiers stay local and are never printed.
    """
    icu_to_subject = {}
    with Path(demog_csv).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        for row in reader:
            icu = int(float(row["icustay_id"]))
            subject = int(float(row["subject_id"]))
            if icu in icu_to_subject and icu_to_subject[icu] != subject:
                raise ValueError("ambiguous demographic ICU-to-subject mapping")
            icu_to_subject[icu] = subject

    targets = {}
    target_slices = {}
    for name, data in [("train", train), ("test", test)]:
        target_slices[name] = episode_slices(data["done"])
        for ep, rows in enumerate(target_slices[name]):
            fingerprint = trajectory_fingerprint(*(data[f][rows] for f in
                ["state", "next_state", "action", "reward", "done"]))
            targets.setdefault(fingerprint, []).append((name, ep))

    # csv clinical-note fields can exceed the Python default field-size limit.
    csv.field_size_limit(64 * 1024 * 1024)
    matches = {}
    source_rows = 0
    source_episodes = 0

    def consume(rows):
        nonlocal source_episodes
        source_episodes += 1
        rows = sorted(rows, key=lambda row: int(float(row["step"])))
        if len(rows) <= 1:
            return
        icus = {int(float(row["m:icustayid"])) for row in rows}
        if len(icus) != 1:
            raise ValueError("a source trajectory spans multiple ICU stays")
        state = np.asarray([[float(row[col]) for col in state_columns] for row in rows], dtype=np.float32)
        action = np.asarray([int(float(row["a:action"])) for row in rows[:-1]], dtype=np.int64)
        reward = np.asarray([float(row["r:reward"]) for row in rows[1:]], dtype=np.float32)
        done = (reward != 0).astype(np.uint8)
        fingerprint = trajectory_fingerprint(state[:-1], state[1:], action, reward, done)
        if fingerprint in targets:
            icu = next(iter(icus))
            if icu not in icu_to_subject:
                raise ValueError("a matched source ICU stay has no subject mapping")
            matches.setdefault(fingerprint, []).append(icu_to_subject[icu])

    with Path(cohort_csv).open(newline="") as stream:
        reader = csv.DictReader(stream)
        # The original notebook selects membership in source-header order,
        # concatenating observations first and demographics second.
        state_columns = [c for c in reader.fieldnames if c in OBS_COLUMNS]
        state_columns += [c for c in reader.fieldnames if c in DEM_COLUMNS]
        if len(state_columns) != 43:
            raise ValueError("source cohort must contain the verified 43 features")
        current = None
        rows = []
        seen = set()
        for row in reader:
            source_rows += 1
            trajectory = row["traj"]
            if current is not None and trajectory != current:
                consume(rows)
                seen.add(current)
                rows = []
            if trajectory in seen:
                raise ValueError("source CSV trajectories are not contiguous")
            current = trajectory
            # Drop note/timestamps from memory; only selected numeric fields matter.
            rows.append({key: row[key] for key in state_columns +
                         ["step", "m:icustayid", "a:action", "r:reward"]})
        if rows:
            consume(rows)

    groups = {name: np.empty(len(data["state"]), dtype=np.int64)
              for name, data in [("train", train), ("test", test)]}
    for fingerprint, entries in targets.items():
        subjects = matches.get(fingerprint, [])
        # Multiple identical source episodes are acceptable only if all identify
        # the same subject; otherwise leakage filtering would be ambiguous.
        if not subjects or len(set(subjects)) != 1:
            raise ValueError("missing or ambiguous complete-trajectory subject match")
        for name, ep in entries:
            groups[name][target_slices[name][ep]] = subjects[0]
    train_subjects = np.unique(groups["train"])
    test_subjects = np.unique(groups["test"])
    overlap = np.intersect1d(train_subjects, test_subjects)
    keep = ~np.isin(groups["train"], test_subjects)
    report = {
        "mode": "verified_subject", "method": "SHA256 of full float32 state/next_state/action/shifted_reward/done trajectory",
        "cohort": {"path": str(Path(cohort_csv).resolve()), "sha256": sha256_file(cohort_csv)},
        "demographics": {"path": str(Path(demog_csv).resolve()), "sha256": sha256_file(demog_csv)},
        "feature_columns": state_columns, "source_rows": source_rows,
        "source_episodes": source_episodes,
        "matched_train_episodes": len(target_slices["train"]),
        "matched_test_episodes": len(target_slices["test"]),
        "train_subjects_before_filter": len(train_subjects), "test_subjects": len(test_subjects),
        "overlapping_subjects": len(overlap), "excluded_train_rows": int((~keep).sum()),
        "excluded_train_episodes": sum(not keep[rows.start] for rows in target_slices["train"]),
        "train_subjects_after_filter": len(np.unique(groups["train"][keep])),
        "original_test_untouched": True,
    }
    return groups["train"], groups["test"], keep, report


DEFAULT_RAW_COHORT = ORIGINAL / "dataset/mimic3/sepsis_final_data_RAW_withTimes.csv"


_METADATA = ("traj", "step", "m:charttime", "m:icustayid", "a:action", "r:reward")


_FIELDS = ("state", "next_state", "action", "reward", "done")


def _integer(value, field):
    number = float(value)
    if not np.isfinite(number) or number != int(number):
        raise ValueError("source " + field + " must contain finite integers")
    return int(number)


def _validate_split(data):
    if any(field not in data for field in _FIELDS):
        raise ValueError("each split requires all five transition fields")
    state, _ = validate_data(data["state"], data["action"], data["done"])
    next_state = np.asarray(data["next_state"])
    reward = np.asarray(data["reward"]).reshape(-1)
    done = np.asarray(data["done"]).reshape(-1)
    if state.shape[1] != 43 or next_state.shape != state.shape:
        raise ValueError("state and next_state must be aligned N x 43 arrays")
    if len(reward) != len(state) or not np.isfinite(next_state).all():
        raise ValueError("next_state/reward dimensions or finiteness disagree")
    if not np.isfinite(reward).all() or not np.isin(reward, [-1, 0, 1]).all():
        raise ValueError("Sepsis reward must be zero or terminal +/-1")
    if not np.array_equal(done, reward != 0):
        raise ValueError("Sepsis done must equal the terminal reward indicator")
    return episode_slices(done)


def _source_episodes(path, columns=None):
    """Yield (trajectory key, numeric episode, selected feature columns).

    One CSV trajectory is held at a time.  Clinical-note strings are discarded
    as soon as the reader yields a row.  CSV trajectory order need not agree
    with the saved train/test arrays, and steps are explicitly sorted.
    """
    csv.field_size_limit(64 * 1024 * 1024)
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError("duplicate source CSV columns")
        selected = [col for col in header if col in OBS_COLUMNS]
        selected += [col for col in header if col in DEM_COLUMNS]
        if len(selected) != 43 or any(key not in header for key in _METADATA):
            raise ValueError("source CSV must contain 43 features and trajectory metadata")
        if columns is None:
            columns = selected
        elif set(selected) != set(columns):
            raise ValueError("RAW and normalized feature columns disagree")
        current = None
        rows = []
        seen = set()
        for row in reader:
            key = row["traj"]
            if current is not None and key != current:
                yield current, _numeric_episode(rows, columns), columns
                seen.add(current)
                rows = []
            if key in seen:
                raise ValueError("source CSV trajectories are not contiguous")
            current = key
            rows.append({col: row[col] for col in list(columns) + list(_METADATA)})
        if rows:
            yield current, _numeric_episode(rows, columns), columns


def _numeric_episode(rows, columns):
    rows = sorted(rows, key=lambda row: _integer(row["step"], "step"))
    step = np.asarray([_integer(row["step"], "step") for row in rows], dtype=np.int64)
    if not np.array_equal(step, np.arange(len(rows))):
        raise ValueError("source steps must be unique and consecutive from zero")
    icu = np.asarray([_integer(row["m:icustayid"], "ICU identifier") for row in rows])
    if len(np.unique(icu)) != 1:
        raise ValueError("source trajectory spans multiple ICU stays")
    charttime = np.asarray([float(row["m:charttime"]) for row in rows], dtype=np.float64)
    state = np.asarray([[float(row[col]) for col in columns] for row in rows], dtype=np.float64)
    action = np.asarray([_integer(row["a:action"], "action") for row in rows], dtype=np.int64)
    reward = np.asarray([float(row["r:reward"]) for row in rows], dtype=np.float64)
    if not np.isfinite(state).all() or not np.isfinite(charttime).all():
        raise ValueError("non-finite source states or times")
    if (np.diff(charttime) < 0).any():
        raise ValueError("source times decrease within a trajectory")
    if not ((action >= 0) & (action < 25)).all():
        raise ValueError("source actions must be in [0, 24]")
    if not np.isin(reward, [-1, 0, 1]).all():
        raise ValueError("source rewards must be zero or terminal +/-1")
    if len(rows) > 1 and (reward[:-1] != 0).any():
        raise ValueError("source reward must occur only on the final source row")
    if len(rows) > 1 and reward[-1] == 0:
        raise ValueError("source trajectory has no terminal reward")
    return {"state": state, "step": step, "charttime": charttime,
            "icu": icu, "action": action, "reward": reward}


def _transitions(episode):
    reward = episode["reward"][1:]
    return {"state": episode["state"][:-1], "next_state": episode["state"][1:],
            "action": episode["action"][:-1], "reward": reward,
            "done": (reward != 0).astype(np.uint8)}


def recover_raw_states(train, test, normalized_csv=DEFAULT_COHORT,
                       raw_csv=DEFAULT_RAW_COHORT):
    """Return ({split: {state, next_state}}, JSON-safe provenance report).

    Call this on full, original done-delimited arrays before filtering patients
    or splitting trajectories.  A normalized fingerprint match alone is not
    enough: duplicate source matches fail, since clipping could conceal distinct
    physical states.  No scaler is fitted here.  Fit a new scaler on the intended
    train state rows only after patient partitioning.
    """
    started = time.monotonic()
    data = {"train": train, "test": test}
    slices = {name: _validate_split(split) for name, split in data.items()}
    targets = {}
    for name, split in data.items():
        for index, rows in enumerate(slices[name]):
            fingerprint = trajectory_fingerprint(*(split[field][rows] for field in _FIELDS))
            targets.setdefault(fingerprint, []).append((name, index, rows))

    matched_sources = {}
    matched_fingerprints = set()
    columns = None
    source_rows = source_episodes = 0
    gaps = {}
    for key, episode, selected in _source_episodes(normalized_csv):
        columns = selected
        source_episodes += 1
        source_rows += len(episode["step"])
        if len(episode["step"]) <= 1:
            continue
        transition = _transitions(episode)
        fingerprint = trajectory_fingerprint(*(transition[field] for field in _FIELDS))
        if fingerprint not in targets:
            continue
        if fingerprint in matched_fingerprints:
            raise ValueError("ambiguous normalized trajectory match; physical states may differ")
        matched_fingerprints.add(fingerprint)
        # The float32 digest verifies source current/next states and shifted
        # outcomes; retain only join metadata, never normalized state or notes.
        matched_sources[key] = (fingerprint, {field: episode[field] for field in
            ["step", "charttime", "icu", "action", "reward"]})
        for seconds in np.diff(episode["charttime"]):
            label = str(float(seconds / 3600))
            gaps[label] = gaps.get(label, 0) + 1
    if columns is None or matched_fingerprints != set(targets):
        raise ValueError("missing complete normalized trajectory match")

    result = {name: {field: np.empty(np.asarray(split["state"]).shape, dtype=np.float64)
                     for field in ["state", "next_state"]} for name, split in data.items()}
    matched_raw = set()
    raw_rows = raw_episodes = 0
    for key, episode, _ in _source_episodes(raw_csv, columns):
        raw_episodes += 1
        raw_rows += len(episode["step"])
        if key not in matched_sources:
            continue
        if key in matched_raw:
            raise ValueError("duplicate RAW trajectory match")
        fingerprint, metadata = matched_sources[key]
        for field, expected in metadata.items():
            if not np.array_equal(episode[field], expected):
                raise ValueError("RAW/normalized source " + field + " alignment disagrees")
        transition = _transitions(episode)
        for name, index, rows in targets[fingerprint]:
            for field in ["action", "reward", "done"]:
                if not np.array_equal(np.asarray(data[name][field][rows]).reshape(-1), transition[field]):
                    raise ValueError("reconstructed source " + field + " disagrees with saved arrays")
            result[name]["state"][rows] = transition["state"]
            result[name]["next_state"][rows] = transition["next_state"]
        matched_raw.add(key)
    if matched_raw != set(matched_sources):
        raise ValueError("missing complete RAW trajectory match")
    if raw_rows != source_rows or raw_episodes != source_episodes:
        raise ValueError("RAW and normalized complete-source trajectory/row counts disagree")

    ranges = {}
    for feature, column in enumerate(columns):
        ranges[column] = {"min": min(float(values[field][:, feature].min()) for values in result.values()
                                     for field in ["state", "next_state"]),
                          "max": max(float(values[field][:, feature].max()) for values in result.values()
                                     for field in ["state", "next_state"])}
    report = {
        "method": "full float32 normalized-transition SHA256, then exact RAW trajectory/step join",
        "normalized_cohort": {"path": str(Path(normalized_csv).resolve()), "sha256": sha256_file(normalized_csv)},
        "raw_cohort": {"path": str(Path(raw_csv).resolve()), "sha256": sha256_file(raw_csv)},
        "feature_columns": columns,
        "source_rows": source_rows, "source_episodes": source_episodes,
        "raw_source_rows": raw_rows, "raw_source_episodes": raw_episodes,
        "matched_train_episodes": len(slices["train"]),
        "matched_test_episodes": len(slices["test"]),
        "matched_unique_source_episodes": len(matched_sources),
        "shapes": {name: list(split["state"].shape) for name, split in result.items()},
        "output_dtype": "float64", "all_states_finite": True,
        "feature_ranges_state_and_next_state": ranges,
        "matched_unique_source_transition_gaps_hours": gaps,
        "action_reward_done_alignment_verified": True,
        "raw_normalized_step_time_icu_alignment_verified": True,
        "terminal_next_state": "final RAW source row, preserved; never replaced by current state",
        "source_transition_convention": "state[:-1], next_state=state[1:], action[:-1], reward[1:], done=(reward[1:]!=0)",
        "scaler_fitted": False, "global_normalization_and_clipping_bypassed": True,
        "imputation_leakage_removed": False,
        "limitations": [
            "RAW means already aggregated and imputed physical-unit source states, not original raw events.",
            "Original row-index interpolation used future observations without patient boundaries.",
            "Original KNN imputation used whole-cohort 9999-row chunks before the new patient split.",
            "Original 4-hour bin skipping, inclusive boundaries, within-bin state/action overlap and 90-day terminal outcome are unchanged.",
            "New train-only scaling cannot remove the already applied original imputation leakage or make states causal pre-decision snapshots.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    return result, report


# Historical artifact class-path compatibility

_LOCK = threading.RLock()


@contextmanager
def _historical_behavior_namespace():
    import model as behavior
    # Only serialized historical module names are supported here. No old source
    # module or estimator implementation is imported or left installed.
    parent = types.ModuleType("models")
    parent.__path__ = []
    parent.behavior = behavior
    aliases = {"RL_behavior_retrain": behavior, "models": parent, "models.behavior": behavior}
    missing = object()
    with _LOCK:
        previous = {name: sys.modules.get(name, missing) for name in aliases}
        try:
            sys.modules.update(aliases)
            yield
        finally:
            for name, module in previous.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


def load_behavior_model(path):
    import joblib
    with _historical_behavior_namespace():
        return joblib.load(path)
