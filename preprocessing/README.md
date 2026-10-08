# Clinical preprocessing code

This directory contains code references from clinical preprocessing through replay-buffer generation. Notebook outputs, execution counts, widgets, embedded attachments, raw records, cohort CSVs, patient split manifests, tuples, and NPY buffers are not bundled. The original research copies remain outside this release.

## Heparin

- `heparin/mimic3/`: one-hour cohort and buffer-generation notebooks, including the discrete-action variants used in the study's data lineage. Three-hour variants, exploratory analysis notebooks, and temporary scaling/test variants are omitted.
- `heparin/mimic4/preprocess.ipynb`: the corrected observed-aPTT reward path, supported by `notebook_support.py` and the root `util.py`. The duplicated original IV notebook is omitted.

The historical III notebook mixes an imputed-PTT reward branch with reads of the previously generated `no_reward_SAH` cohort. The actual study buffer used measured aPTT, with reward zero in unmeasured bins. Inclusion of that notebook is provenance, not a claim that rerunning every cell reproduces the original buffer.

For the corrected IV notebook, set `HEPARIN_MIMIC4_SOURCE_ROOT` to the original clinical input directory and optionally set `UNCERTAINTY_RL_ROOT` and `HEPARIN_MIMIC4_OUTPUT_ROOT`. The original saved split manifest is a local input; it is intentionally not distributed because it contains patient/stay membership. The notebook creates a fresh output directory and does not silently replace the original split.

## Sepsis

- `sepsis/cohort_preprocessing/mimic3/`: clinical extraction, cohort construction, and trajectory splitting.
- `sepsis/cohort_preprocessing/mimic4/`: extraction, clinical feature helpers, MDP construction, and trajectory splitting.
- `sepsis/mimic3_split_cohort.ipynb`, `mimic4_split_cohort.ipynb`, and `util.py`: original buffer-generation references. Mixed clinical-note processing remains where it occurs in the same notebook.

Database extraction notebooks read the connection string from `MIMIC_DATABASE_DSN`. Prepare your local database and raw files first. Optional note-processing cells require their own locally installed dependencies and inputs.

The duplicated MIT-LCP `mimic-code` checkout is omitted; install the upstream database definitions separately. Non-vendored SQL setup code is retained where present. Integrity-check notebooks are omitted because they inspect saved local products rather than generate clinical buffers.

Install optional notebook dependencies with `python -m pip install -r preprocessing/requirements.txt`. These are historical pipelines with local file assumptions, not a fully automated raw-to-buffer command.

Optional Hugging Face authentication reads `HF_TOKEN` from the environment; no literal access tokens are distributed.
