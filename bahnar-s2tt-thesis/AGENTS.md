# Project Context — Bahnar S2TT Thesis

## Project

Master's thesis on low-resource Bahnar-to-Vietnamese Speech Translation.

Research aim:
Evaluate and improve Bahnar speech-to-Vietnamese text translation under low-resource conditions.

## Research Questions

RQ1:
Compare a Cascaded ASR + MT system with a Direct Speech-to-Text Translation system.

RQ2:
Using the same Direct S2TT architecture and equal data budgets, compare:
1. randomly selected pseudo-labelled training data;
2. pseudo-labelled data selected by quality score.

The chatbot, workflow and RAG components are application/demo layers.
They are not the main research contribution.

## Dataset

Primary dataset:
`cuong06/Bahnar_Vietnamese`

Audited row counts:

- Train: 113,830 rows.
- Original validation: 2,636 rows.
- Original test: 3,336 rows.
- Validation with public audio reference: 205 rows.
- Test with public audio reference: 10 rows.

Current no-human-data plan:

- Split the original 113,830-row train set into a new train set and a new validation set.
- Combine the 205 accessible validation rows and 10 accessible test rows into a fixed 215-row public test candidate.
- Rows with `audio=None` are excluded from the primary speech experiment.
- Do not use the 215 test samples for model selection or hyperparameter tuning.

## Dataset Owner Clarification

The data comes mainly from:

- Gia Lai TV;
- VTV5/YouTube;
- Oneway Radio.

Missing validation/test audio is caused by redistribution restrictions.
It cannot be recovered through Hugging Face authentication or a PRO account.

The `speaker_id` field may represent a source or programme.
For example, `speaker_id=oneway` refers to Oneway Radio's Bahnar programme.
It is not a verified biometric speaker identity.

Therefore:

- Do not report `unique speaker_id` as the number of real speakers.
- Treat the field as `source_label` unless verified otherwise.
- Do not claim speaker-independent evaluation from this field alone.

## Data Splitting Rules

`recording_group_id` is used only for safe data splitting.
It is not an input feature for the model.

Segments originating from the same video, programme, episode or recording
must remain in the same split.

Preferred grouping order:

1. explicit video or recording ID;
2. episode/session ID;
3. meaningful prefix derived from `audio_path`;
4. meaningful prefix derived from `record_id`;
5. source label only as a low-confidence fallback.

Do not randomly split individual rows when multiple rows may come from
the same original recording.

Target split:

- approximately 90% train;
- approximately 10% validation;
- validation should preferably stay within 8–12%;
- group overlap between train and validation must be zero.

## Leakage Rules

Required checks:

- train/validation recording-group overlap = 0;
- train/test exact record overlap = 0;
- train/test audio-key overlap = 0;
- train/test exact Bahnar–Vietnamese pair overlap = 0;
- validation/test overlap = 0.

Do not inspect test results repeatedly while developing models.

## Notebook 01

Primary file:

`notebooks/01_data_audit_and_split.ipynb`

Notebook 01 must:

1. audit row counts and schema;
2. audit audio-reference and text availability;
3. treat `speaker_id` as a source label;
4. derive and review `recording_group_id`;
5. check duplicate rows and leakage;
6. verify accessible evaluation audio;
7. create a recording-aware train/validation split;
8. create the fixed public test candidate;
9. export immutable manifests and a split summary.

Expected outputs:

- `rq1_train.csv/parquet`
- `rq1_validation.csv/parquet`
- `rq1_test.csv/parquet`
- `split_summary.json`

Do not proceed to model training until Notebook 01 passes all integrity checks.

## Planned Notebooks

1. Data audit and split.
2. Cascaded ASR training.
3. Cascaded MT training.
4. Direct S2TT training.
5. RQ1 evaluation.
6. Pseudo-label quality scoring.
7. RQ2 controlled experiments.

## Working Constraints

- Total project plan: 12 weeks.
- Training environment: Google Colab.
- Persistent artifacts: Google Drive.
- Source code and version control: local IDE and GitHub.
- Prefer reusable functions in `src/` instead of duplicating long code in notebooks.
- Explain technical findings in Vietnamese because the researcher is still learning AI.

## Current Task

Review Notebook 01 first.

Do not modify files unless explicitly requested.
Report findings before proposing or applying fixes.