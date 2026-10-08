# Offline RL and OPE on FrozenLake

The scripts construct an offline FrozenLake experiment and compute DR, WDR, WIS, FQE, and importance-weight ESS. Clinical input files are not needed. Generated trajectories, checkpoints, figures, and historical results are not included.

```bash
python -m pip install -r requirements_ope.txt -r gym_test/requirements.txt
python -m gym_test.run_experiment --output-dir gym_test/results/new_run
python -m gym_test.verify_results --run-dir gym_test/results/new_run
```

To replace the known behavior denominator with the same Random Forest behavior-cloning procedure used for the clinical experiments, run `python -m gym_test.run_bc_experiment --help` after creating the base experiment. `verify_bc_results.py` and `audit_calibration.py` check the saved products of that run.

Bootstrap intervals condition on the frozen target policies, behavior models, and critics. ESS diagnoses importance-weight concentration and does not by itself validate clinical support.
