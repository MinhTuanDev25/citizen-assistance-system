# Bahnar → Vietnamese S2TT Thesis (temporary research workspace)

Scaffold for RQ1 (Cascaded vs Direct). Application chatbot stays outside this folder.

## Layout

```text
bahnar-s2tt-thesis/
├── notebooks/          # 01 audit → 06 evaluation
├── src/                # shared helpers
├── configs/            # asr / mt / direct
├── data/manifests/     # locked splits from Notebook 01
├── data/audit/         # audit CSVs
├── checkpoints/
├── predictions/
├── metrics/
└── results/
```

## Environment (important)

Do **not** install into a shared Anaconda/base env (often has `gym` + `dask` fighting over `cloudpickle`).

```bash
cd bahnar-s2tt-thesis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=bahnar-s2tt --display-name="Python (bahnar-s2tt)"
```

In the notebook: **Select Kernel → Python (bahnar-s2tt)**.

## Start here

1. Open `notebooks/01_data_audit_and_split.ipynb` with kernel `bahnar-s2tt`
2. Keep `GROUP_REVIEW_APPROVED=False` on first run
3. Review audit outputs under `data/audit/`
4. Then approve split and continue with Notebook 02+

Notebooks 02–06 are placeholders until manifests exist.
