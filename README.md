# Uncertainty-aware offline reinforcement learning

Offline reinforcement learning and off-policy evaluation for the MIMIC-III heparin and sepsis cohorts, with clinical preprocessing references for MIMIC-III/IV.

## Code layout

| File or directory | Purpose |
|---|---|
| `agent.py` | DQN, DDQN, BCQ, CQL, behavior estimation, and frozen policy probabilities |
| `model.py` | Policy networks, FQE critics, and behavior-model wrappers |
| `metric.py` | DR, WDR, WIS, FQE, bootstrap intervals, ESS, and OOD metrics |
| `util.py` | Replay buffers, patient splits, preprocessing helpers, paths, and artifact loading |
| `detector.py`, `plot.py` | OOD detection and plotting |
| `configs/`, `scripts/` | Configuration and experiment entry points |
| `preprocessing/` | Clinical preprocessing and buffer-generation notebooks and helpers |
| `tests/` | Tests using synthetic inputs and temporary files |
| `gym_test/` | FrozenLake experiments with known and behavior-cloned action probabilities |

## Installation and tests

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements_ope.txt` records the versions used for the existing CPU OPE runs. `requirements.txt` adds training/configuration and embedding dependencies. The preprocessing notebooks and Gym experiments have separate requirements files.

## Local clinical inputs

Place the following arrays under `dataset/heparin/` or `dataset/sepsis/` for each `train` and `test` split:

```text
<split>_state.npy
<split>_next_state.npy
<split>_action.npy
<split>_reward.npy
<split>_done.npy
```

MIMIC-III heparin uses 16 state features and 6 actions; sepsis uses 43 features and 25 actions. Patient separation also requires local cohort/demographics inputs. The behavior-fitting CLI accepts `--cohort-csv` and `--demog-csv`; the generic training entry point uses the default paths resolved by `prepare_subject_split`. The loader converts historical heparin `done=0` terminal arrays to the internal `done=1` terminal convention. Stored scalar `BC_prob` arrays are not used by current OPE.

Clinical records, patient identifiers, generated buffers, models, and historical experiment outputs are kept outside this code distribution. Preprocessing notebooks have no execution outputs. Their database credentials and raw-input locations must be supplied locally.

## Training and behavior estimation

```bash
python -m scripts.train_policy --dataset sepsis --algorithm CQL --output-dir outputs/new_sepsis_run
python -m scripts.fit_behavior --dataset sepsis --cohort-csv /path/to/cohort.csv --demog-csv /path/to/demographics.csv --output-dir outputs/new_behavior
```

Run a command with `--help` for its supported options. `train_policy` and `fit_behavior` operate on local clinical inputs; historical tuning, comparison, and reporting scripts additionally require their documented local `outputs/` artifacts. Those artifacts are not bundled. The generic training defaults are separate from the settings used for the final manuscript comparison.

## OPE conventions

Policy probabilities are explicitly fixed as greedy or Q-softmax. DR/WDR use a separately fitted fixed-policy FQE critic. Behavior probabilities are estimated on policy-training patients with a calibrated Random Forest. Sepsis has one terminal reward (+1 for 90-day survival, -1 for death) and zero intermediate rewards; heparin has per-transition rewards.

The final comparison used Q-softmax temperature 1, discount 0.98, and cumulative importance-weight prefix clipping at 5. The numerical API defaults to no clipping. Cumulative clipping applies to each original prefix product; clipped values are never fed into subsequent products. WDR normalizes across trajectories at each time and retains ended trajectories in the denominator using absorbing padding. Bootstrap resamples whole patients, holding the selected policies and fitted nuisance models fixed.

## Preprocessing references

See `preprocessing/README.md`. Historical notebook code is provided as a reference for the original data pipeline; inclusion does not establish that every notebook was rerun from raw records or that it independently reproduces the frozen clinical buffers. MIMIC databases and upstream SQL setup must be installed separately. Relevant upstream implementations include Microsoft `mimic_sepsis` and MIT-LCP `mimic-code`.

## Gym example

```bash
python -m pip install -r gym_test/requirements.txt
python -m gym_test.run_experiment --output-dir gym_test/results/new_run
python -m gym_test.verify_results --run-dir gym_test/results/new_run
python -m gym_test.run_bc_experiment --help
```

## Source verification

`source_manifest.json` contains hashes for this distribution's runtime, tests, and corrected heparin notebook interface. Deliberate source edits require updating that manifest before running scripts that enforce it. Compatibility aliases in `util.load_behavior_model` are used only while loading previously saved models; obsolete modules are not included.
