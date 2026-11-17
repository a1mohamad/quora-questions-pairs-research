# QQP LSTM–attention experiment handoff

Prepared: 2026-09-09. Updated: 2026-10-15 (overall QQP roadmap and model-selection strategy). This document records the discussion, inspected notebook evidence, and proposed experiments for a future session. Creating this document did not implement new models or start training.

## Purpose and working constraints

Explore how attention scoring and the location of question interaction affect Quora Question Pairs duplicate classification while retaining a BiLSTM backbone. Correct implementation matters more to the user than architectural labels.

- Work from the user's existing notebooks and trainer. Prefer focused changes; do not introduce a new trainer, mixin, experiment framework, or broad refactor without a specific reason and agreement.
- Preserve automatic model-architecture export to Python, requirements generation, notebook capture, and MLflow logging. The user explicitly wants automation, not manual copying of class definitions into exported artifacts.
- Preserve existing experiments and outputs. Use separate output directories/run identities for new experiments.
- Resumable training was deferred by the user. Do not incorporate the previously proposed resume helper or new trainer as part of this architecture work by default.
- This document is a handoff, not authorization to launch all experiments unattended. At the start of the next session, inspect current files and follow the user's requested implementation scope.
- Separate verified historical behavior, agreed experiment interests, and recommendations. Do not interpret an old configuration label as proof of executed behavior.

## November 17, 2026 continuation — implementation handoff for the next experiments

**Read this section first when resuming.** This addendum reflects the November QQP notebook/Kaggle work and extends, rather than replaces, the historical audit and ten-experiment roadmap below. The older `Historical status` column and run inventory are **September audit history**, not a current training ledger. The five notebooks supplied with this update are the current references; inspect their actual code on every new iteration. **Do not silently refactor the notebook or change model/trainer code beyond the requested experiment.**

### 0. Current assets and reported status (as of this conversation)

The user supplied these *uploaded copies*; parentheses such as `(7)` and `(2)` are copy suffixes, **not** the intended Kaggle dataset filenames:

| Core ID or variant | Notebook supplied for inspection | Role / status |
|---|---|---|
| **#1**, GloVe independent pooling | `02a-lstm-bahdanau-indepattn-glove-train(1).ipynb` | Whitespace/GloVe baseline. This uploaded notebook uses an older local-file layout; the handoff's historical v9/v10 results are separate recorded runs. Do not assume the uploaded file has Kaggle migration cells. |
| **#2**, WordPiece independent pooling | `02a-lstm-bahdanau-indepattn-wordpiece-train.ipynb` | **Starting point for #4**: actual WordPiece tokenizer/data/embedding/training pipeline, Kaggle inputs and MLflow migration. Its *internal* dataset-copied filename remains `02-LSTM-attention-train.ipynb` despite this uploaded filename; fix for new notebooks. Completion/metrics of a newly trained #2 run have not been verified in this handoff. |
| **#3A**, GloVe additive cross + PAD-only pooling | `02b-lstm-bahdanau-crossattn-glove-train(7).ipynb` | Completed training **reported by user**; no run ID/F1/summary supplied here. Canonical dataset notebook: `02b-lstm-bahdanau-crossattn-glove-train.ipynb`. Use as default GloVe cross-attention reference. |
| **#3B**, GloVe additive cross + stopword-masked pooling | `02b-lstm-bahdanau-crossattn-glove-stopmask-train(2).ipynb` | Ablation run **reported started** on Nov 17; its completion and performance are **not yet verified**. Canonical dataset notebook: `02b-lstm-bahdanau-crossattn-glove-stopmask-train.ipynb`. Keep separate from #3A. |
| **#4** (next) | **Not yet implemented in the supplied files** | WordPiece + trainable-from-scratch embeddings + bidirectional custom Bahdanau cross-attention. Start with **#2's WordPiece notebook**, graft only relevant attention/model changes from **#3A** (not #3B stopword masking unless specifically asked), and carry forward the newer file/ZIP conventions. |

**Two notebook families, one shared MLflow history:**

- **Whitespace/GloVe family (#1, #3, #5, #7, #9):** preserve the word-level cleaning/tokenization and GloVe loader; for new architectures begin with the **latest relevant GloVe-family notebook**, generally #3A for the next non-ablation architecture. #3B is a controlled masking variant, **not** the default base for every subsequent GloVe model.
- **Custom WordPiece/learned-embedding family (#2, #4, #6, #8, #10):** preserve the trained WordPiece tokenizer, prepared `train_split.csv`/`valid_split.csv`, existing preprocessing and WordPiece embedding policy. For #4 **start from #2**; for later WordPiece experiments use the most recent validated WordPiece-family implementation. Do **not** copy the GloVe tokenizer, raw-data split, pretrained vectors, or embedding freeze/unfreeze logic into WordPiece.
- Both families can use the same `Quora-Question-Pairs/LSTM_attention` MLflow experiment in the shared SQLite DB. The **core experiment ID** is different from MLflow `-vN` run versions. Maintain unambiguous `ATTENTION_TYPE`, tokenizer/embedding metadata, notebook filename, output directory, and architecture filename for each version.

### 1. The actual #3A model, and exactly what #3B changes

The current #3A is **not** independent token-to-token self-attention. It is:

```text
Q1 word IDs -> pretrained GloVe(100d) -> shared packed 2-layer BiLSTM -> H1 [B,L1,768]
Q2 word IDs -> pretrained GloVe(100d) -> shared packed 2-layer BiLSTM -> H2 [B,L2,768]
           -> 4-head additive/Bahdanau cross-attention BOTH directions
                  C1 = Cross(Q=H1, K=H2, V=H2)
                  C2 = Cross(Q=H2, K=H1, V=H1)
           -> LayerNorm(H1 + C1), LayerNorm(H2 + C2) (SKIP_CONNECTION=True)
           -> independent 4-head additive ATTENTION POOLING on updated tokens
           -> one 768d summary per question, pair features, MLP -> one logit
```

Cross-attention scores per head are `v^T tanh(Wq h_i + Wk h_j)`, producing `[B,L1,L2]` scores before softmax on Q2's source-token axis. A per-token Q1 context is a weighted sum of Q2's projected **value** vectors; Q1 influences the **weights** via its query, and the residual directly retains Q1's original vector. The reverse direction shares cross-attention module weights. `self.score: Linear(64,1)` returns **one score per token pair**, not one score for the entire tensor. `nn.Linear` applies to the last dimension and preserves the leading `[B,L1,L2]` axes. Batch size `B` counts **question pairs**, not questions (two pairs contain four questions); padding makes lengths regular within each question tensor. The per-head context width is `768/4=192` and concatenation/projection returns 768 dimensions. A token-to-token self-attention block **within each question only** is a different optional ablation; joint self-attention means tokens from both questions share a combined attention sequence. Additive versus scaled dot-product refers to score computation, not to whether attention is self/cross.

The final pooling is **separate**: it assigns learned importance to the updated tokens and compresses `[B,L,768]` into `[B,768]` per question. It is a valid design, but not proven superior to masked mean or mean+max pooling; those alternative pooling ablations remain listed in the original “Additional options discussed” section. Don't call independent attention pooling “token-to-token self-attention.”

**Relevant existing #3A configs** (verify anew when editing): `MODEL_TYPE='LSTM_attention'`, `ATTENTION_TYPE='MultiHead-Bahdanau-Cross-Pool'`, `TOKENIZER_TYPE='Whitespace'`, `EMBEDDING_TYPE='GloVe-6B-100d'`, `EMB_DIM=100`, `HIDDEN_DIM=384` per BiLSTM direction, `NUM_LAYERS=2`, `NUM_HEADS=4`, `ATTENTION_SCORE_DIM=64`, `MAX_LENGTH=50`, `VOCAB_SIZE=20000`, `SKIP_CONNECTION=True`, 2-layer BiLSTM packing, PAD-only masks. `FREEZE_TOKEN_EMBEDDING=True` with `UNFREEZE_EPOCH=3` means **frozen epochs 1–3; unfreeze at the start of epoch 4** (zero-based epoch counter). The trainer was reported to assign an embedding-specific learning rate after unfreezing; verify in `Trainer` rather than inferring it from config alone.

**#3B controlled masking ablation (already implemented in the supplied #3B notebook):**

1. Preserve #3A's BiLSTM, both cross-attention directions, residual/normalization, **cross-attention PAD-only masking**, downstream attention-pooling heads and classifier.
2. In `QuoraTokenizer`, `build_pooling_mask(stop_words, vocab_size)` maps NLTK English stopwords through `self.vocabs` into a boolean lookup mask (`False` for stopwords and PAD; punctuation and UNK remain `True`). Run it only **after** `tokenizer.build_vocabs(train_df)`, using `stopwords.words('english')`. NLTK's list includes potentially significant terms like `not` and `no`; the ablation tests that choice rather than declaring it correct.
3. `QuoraSiameseClassifier(..., pooling_keep_mask=pool_keep_mask)` calls `register_buffer('pooling_keep_mask', ...)`; the buffer moves with the model and is saved in `state_dict`. At final `_pool(hidden, mask, tokens)`, use `mask & self.pooling_keep_mask[tokens]` and **fall back to PAD-only** when every valid token in that question would otherwise be masked. `forward()` passes `q1`/`q2` token IDs to `_pool`. The original #3A checkpoint and new #3B checkpoint have different `state_dict` structure; load each with its corresponding architecture.
4. #3B `ATTENTION_TYPE='MultiHead-Bahdanau-Cross-Pool-StopMask'`, `ARTIFACT_DIR=.../artifacts/lstm_bahdanau_crossattn_glove_stopmask`, architecture script `02b_stopmask_model_architecture.py`, and its own notebook filename. **Do not overwrite #3A**. In #3A architecture the file is `02b_model_architecture.py` and artifacts are `.../lstm_bahdanau_crossattn_glove`.
5. The first test masks stopwords **during final attention pooling only**. Masking stopwords as cross-attention keys/values, additionally masking punctuation or UNK, and trying masked mean / mean+max pooling are **separate** future ablations. The historical #1 baseline used a broader stopword + punctuation + PAD + UNK pooling mask; do not claim #3B reproduces that exact policy.

**Preprocessing cleanup:** The GloVe #3A/#3B `QuoraPreproccesor` now uses a class-level `_PUNCTUATION = str.maketrans({...})`, then `text.translate(self._PUNCTUATION)` and whitespace splitting/joining, replacing repetitive `re.sub` calls. Maintain the precise mappings (including curly/ASCII dash, fullwidth `？`, quotes, slash and backslash) and the existing empty-question filtering. Do **not** transplant this whitespace preprocessor into the WordPiece pipeline unless explicitly requested; WordPiece already has its own data processing. Historical #1 still displays the original regex style in the supplied notebook.

### 2. Kaggle storage, MLflow migration, and return ZIP — preserve this workflow

**This is a migration of the *existing* SQLite tracking history, not a new MLflow database or an entirely new trainer.** The user explicitly rejected elaborate extra helpers, automatic artifact discovery frameworks, redesigned trainers, fabricated notebook-output captures and unrelated refactors. Maintain the existing simple setup and migration cells, adapting **only source assets, paths and experiment identity**.

**A. Input datasets and writable working directory**

| Family | Input assets used by the supplied notebook | Paths/data policy |
|---|---|---|
| Whitespace + GloVe #3A/#3B | User's versioned Kaggle **Whitespace Training Assets** dataset, official QQP competition input, separate Stanford GloVe 6B 100d dataset | Source assets dataset path in #3A/#3B: `/kaggle/input/datasets/amirmohamadaskari/quera-questions-pairs-whitespace-training-assets` (note actual source spells `quera`; verify exact mounted slug). Source contains the current canonical `.ipynb`, `requirements.py`, `mlflow-quora-questions-pairs.db`. QQP input filename: `train.csv.zip` (the `pd.read_csv` call can read this ZIP if it contains a single CSV). GloVe filename: `glove.6B.100d.txt`. The `attached_file(filename)` helper searches `/kaggle/input` recursively and intentionally errors unless **exactly one** matching file exists; this can fail if multiple inputs contain `train.csv.zip` or the GloVe filename. |
| WordPiece #2 | User's versioned **WordPiece Training Assets** dataset | Source path in #2: `/kaggle/input/datasets/amirmohamadaskari/quora-question-pairs-wordpiece-training-assets`. Its dataset contains `data/processed/train_split.csv`, `data/processed/valid_split.csv`, `artifacts/tokenizers/wordpiece_uncased_30k/`, `requirements.py`, source notebook and SQLite DB. Copies prepared split/tokenizer *folders* into the writable project before training. Does **not** need GloVe or official competition ZIP input. |

Both setup cells copy `mlflow-quora-questions-pairs.db` from the read-only dataset to `/kaggle/working/qqp/mlflow-quora-questions-pairs.db` **only if the working DB does not already exist**. For a new Kaggle run/version, verify the working directory is fresh or that it contains the **latest imported database**, not a previous session's stale copy. Copy the exact canonical notebook + `requirements.py` into `/kaggle/working/qqp`, `os.chdir(WORK_ROOT)`, and add it to `sys.path`; source-copy assertions should succeed. Keep old runs/files in the user's project and do not overwrite historical result folders. The GloVe notebooks build a 90/10 stratified train/validation split from official raw QQP with seed 28; WordPiece loads prebuilt splits. **Do not claim comparable splits unless verified against the underlying row IDs and preprocessing.**

**B. Before `MLflowTracker` initialization:** In the existing SQL migration cell, look up the experiment by `Quora-Question-Pairs/{model_cfg.MODEL_TYPE}` in the imported SQLite database. Save `{experiment_id, windows_root, kaggle_root}` in `mlflow-path-map.json`, and temporarily change that experiment's `artifact_location` to `(path_cfg.MLFLOW_DIR/'Quora-Question-Pairs'/model_cfg.MODEL_TYPE).as_uri()`. The code checks for missing/mismatched prior map state; preserve these guards. Keep the shared `MLflowTracker`, `Trainer`, `EarlyStopping`, `TrainingHistory`, logging, checkpoint and requirements generation code wherever possible. #2 and #3 use the **same `MLflowTracker` source** but their `Trainer` and `TrainConfig` are not byte-identical; preserve family-specific training code instead of blind copy/paste.

**C. After `trainer.fit(...)` completes:** The final Kaggle cell restores the experiment's `artifact_location` to `windows_root`, and rewrites **only URI fields beginning with this session's Kaggle artifact root** for `runs.artifact_uri`, `logged_models.artifact_location`, `model_versions.source`, `model_versions.storage_location` **if the columns exist**. That preserves existing runs' artifact locations. Then use SQLite's `.backup()` API to make a consistent export snapshot, and build `/kaggle/working/qqp-transfer.zip` with `zipfile.ZipFile` containing:

```text
qqp-transfer.zip
├── mlflow-quora-questions-pairs.db      # SQLite backup, paths restored for Windows
├── <unique-model-architecture>.py      # explicit archive.write(path_cfg.MODEL_SCRIPT, ...)
├── mlruns/...                          # new MLflow run artifacts from this Kaggle session
└── artifacts/<unique-model-family>/... # checkpoints, history, configs and other generated artifacts
```

**Important actual-file difference:** #3A and #3B final ZIP cells explicitly include `path_cfg.MODEL_SCRIPT` at ZIP root and already log it as a run artifact. The supplied **WordPiece #2 ZIP cell does NOT yet add the architecture at ZIP root**: it packs the DB + `mlruns/` + `path_cfg.MODEL_ARTIFACT_DIR/`, so **add the explicit `archive.write(path_cfg.MODEL_SCRIPT, arcname=path_cfg.MODEL_SCRIPT.name)` to #4**. Use the family-correct artifact directory when zipping: GloVe #3 uses `ARTIFACT_DIR`; WordPiece #2 uses `MODEL_ARTIFACT_DIR`. Don't zip the shared trained tokenizer as if it were a newly trained model and don't delete it during cleanup. Validate that restored artifact URIs point to where the ZIP will actually be extracted on Windows.

**D. After Kaggle training:** Download/extract the ZIP into the correct local project locations and update the **versioned Kaggle assets dataset** with the new DB for the next run. Kaggle `/kaggle/input` is **read-only**, so no notebook or DB updates are written back to that dataset automatically. The notebook source copied from the assets dataset is **a static source file**, not the live, fully executed Kaggle notebook. The existing notebook artifact capture may contain old saved cell outputs and a prior run ID; the user **rejected** replacing those with synthetic outputs or a fake executed notebook. Use Kaggle **Save Version → Save & Run All** to preserve the genuine executed notebook separately, download that notebook, and (optionally) manually place it under the correct MLflow `<run_id>/artifacts/` after transfer. **Do not represent the dataset snapshot as the live executed notebook.** Keeping the original source and genuine executed notebook separately is preferable. If the last export cell fails or is skipped, the copied DB may still point to Kaggle URIs: inspect and repair safely from backed-up state before use on Windows.

**E. MLflow version compatibility lesson:** In the November session the user reported Kaggle MLflow **3.18.0**, Windows MLflow **3.16.1**; the imported DB had Alembic revision `f4a9b2c7d1e6` while Windows expected `b7e2c1a4d9f3`. `mlflow db upgrade` in 3.16.1 failed with `Can't locate revision ...`; SQLite `PRAGMA integrity_check` returned `ok` and a read-only query counted **54 runs at that time** (not a live current count). Recommended: back up the DB, use matching MLflow versions (3.18.0 on Windows and Kaggle as reported), verify with `mlflow --version` and `mlflow ui --backend-store-uri sqlite:///mlflow-quora-questions-pairs.db --port 5000`, and **do not** manually stamp the Alembic revision or upgrade blindly. The source notebooks install unpinned `mlflow`; pin/check compatible versions in future runs **only with user agreement**. The exact historical row count is not a new run inventory.

### 3. Architecture export, naming, requirements, and provenance

**New established naming pattern:** A standalone notebook for each core experiment or ablation, distinct local artifacts and MLflow run names, with architecture derived automatically from the notebook's canonical filename (not handwritten copies of all model classes):

| Variant | Dataset notebook basename | Architecture export |
|---|---|---|
| #3A PAD-only | `02b-lstm-bahdanau-crossattn-glove-train.ipynb` | `02b_model_architecture.py` |
| #3B stopword mask | `02b-lstm-bahdanau-crossattn-glove-stopmask-train.ipynb` | `02b_stopmask_model_architecture.py` |
| #4 WordPiece cross (proposed, not yet made) | `02b-lstm-bahdanau-crossattn-wordpiece-train.ipynb` | **Use a unique name** such as `02b_wordpiece_model_architecture.py`; plain `02b_model_architecture.py` would collide with the GloVe #3A filename when extracted to the same local project root. |

#3A generates its name in `PathConfig` as:

```python
NOTEBOOK_PATH = ROOT_DIR / '02b-lstm-bahdanau-crossattn-glove-train.ipynb'
MODEL_SCRIPT = ROOT_DIR / f"{NOTEBOOK_PATH.stem.split('-')[0]}_model_architecture.py"
```

#3B intentionally adds `_stopmask_` to the architecture filename. For new experiments use a deterministic but **unique across both tokenizer families** name, e.g. `{prefix}_{tokenizer-or-variant}_model_architecture.py`; keep names consistent with the notebook's automatic exporter and final ZIP's `MODEL_SCRIPT` path. The source notebook `NOTEBOOK_PATH`, setup `for filename in [...]`, notebook title and MLflow run identity must **all** agree. Existing `export_model_from_notebook(path_cfg.NOTEBOOK_PATH, path_cfg.MODEL_SCRIPT)` extracts named class cells from the **copied file on disk**, and `PathConfig.update_requirements()` uses `%updatereqs` on that same file. If the dataset's source notebook is stale, architecture and requirements exports can be stale **even if the live notebook trained newer code**. Republish the latest source into the dataset *before* the Kaggle run, then compare exported architecture class definitions with executed source and do a smoke test. No automatic rewrite of live notebook cell outputs is implemented or desired.

**Export dependency trap:** The #3 cross architecture code uses `pack_padded_sequence` and `pad_packed_sequence`, `model_cfg = ModelConfig()`, `token_cfg = TokenConfig()`. When manually archiving or loading an exported standalone `.py`, ensure those imports and config instances exist **before** the class definitions; otherwise it may import/fail despite otherwise-correct attention code. The relevant model classes for #3 are `AdditiveCrossAttentionHead`, `MultiHeadCrossAttention`, `AttentionHead`, `MultiHeadAttention`, `QuoraSiameseClassifier`, plus configs. The #3B checkpoint additionally contains the `pooling_keep_mask` buffer. Validate forward and backward on a small padded two-question-pair batch, then verify a checkpoint loads into the corresponding exported class without renaming key modules. Historical manual fix: the original log had omitted an architecture file in the transfer ZIP; now the #3 ZIP root has it as well as the MLflow artifact copy.

**Keep MLflow names concise:** Examples actually present in supplied notebooks: #3A `MultiHead-Bahdanau-Cross-Pool`, #3B `MultiHead-Bahdanau-Cross-Pool-StopMask`, #2 `MultiHead-Bahdanau` with `TOKENIZER_TYPE='Wordpiece'`. `MODEL_TYPE='LSTM_attention'` remains unchanged, preserving the shared experiment. Avoid renaming old runs or claiming all 10 names are already implemented. For future experiments label the scoring, interaction and pooling accurately, but don't create sprawling names; record the tokenizer/embedding source separately.

### 4. Next task: how to implement Experiment #4 without repeating migration mistakes

**Target #4:** WordPiece + learned embeddings + **bidirectional custom additive Bahdanau cross-attention after independent BiLSTM encodings**, then final attention pooling and the existing Siamese comparison MLP. The planned architecture is the **WordPiece counterpart of #3A**; do **not** default to #3B's StopMask. This is the next core experiment; no #4 training or complete implementation has been established in this session.

1. **Select the latest verified WordPiece #2 notebook as the base** (`02a-lstm-bahdanau-indepattn-wordpiece-train.ipynb` supplied). In its *actual code*, inspect WordPiece tokenizer loading (`PreTrainedTokenizerFast`), pretrained tokenizer directory, prepared split files, learned embeddings, `TokenConfig` (`VOCAB_SIZE=30000`, `MAX_LENGTH=64`), `QuoraDataset`, and its own trainer/scheduler/checkpoint behavior. Preserve these rather than copying GloVe #3's tokenizer and preprocessing. Also inspect whether its BiLSTM recurrence padding behavior needs an explicitly agreed correction; #2 currently does **not** use packed sequences while #3A does, so changing packing along with cross-attention introduces an additional experimental variable. Document the decision.
2. **Import/adapt only the architecture changes from #3A**: additive cross-attention head(s), 4-head wrapper, both directions `Cross(H1,H2)` and `Cross(H2,H1)`, masks with clearly defined `True=valid` semantics, correct residual/normalization decision, and final pooling/classifier shape handling. WordPiece special IDs and padding handling must come from #2's tokenizer/config, **not** #3A's whitespace hard-coded IDs. Use PAD-only as starting attention policy unless a separate masking ablation is requested.
3. **Keep #2's Kaggle input family**: the WordPiece assets dataset, precomputed train/valid CSVs and tokenizer assets copied into `/kaggle/working/qqp`. Do not attach GloVe or raw QQP competition input merely because #3 used them. Preserve existing MLflow migration cells, `MLflowTracker`, `Trainer`, `TrainingHistory`, checkpoint/config generation and notebook requirements export. Avoid creating a new framework or trainer. Where #2 and #3 trainer differ, verify before porting **any** line.
4. **Give #4 new identities and paths:** canonical notebook filename (proposed) `02b-lstm-bahdanau-crossattn-wordpiece-train.ipynb`; update the **dataset setup copy list** and `PathConfig.NOTEBOOK_PATH` together (the supplied #2 still requests `02-LSTM-attention-train.ipynb` internally). Use a unique run-specific `MODEL_ARTIFACT_DIR`, unique `MODEL_SCRIPT` (e.g. `02b_wordpiece_model_architecture.py`), and descriptive `ATTENTION_TYPE`. Keep `MODEL_TYPE='LSTM_attention'` and WordPiece `TOKENIZER_TYPE`/embedding provenance. Keep shared pretrained tokenizer storage separate from per-run outputs. Do not delete/move prior checkpoint/history directories.
5. **Carry over #3's complete ZIP convention:** add the missing explicit `archive.write(path_cfg.MODEL_SCRIPT, arcname=path_cfg.MODEL_SCRIPT.name)` to #4's final export cell, retain WordPiece's `MODEL_ARTIFACT_DIR` archive loop, SQLite snapshot and URI restoration, the MLflow artifacts loop, and `/kaggle/working/qqp-transfer.zip`. Verify both presence of the exported `.py` and compatibility with checkpoint in a small CPU/GPU smoke test before long training.
6. **Preflight and postflight:** inspect exact Kaggle input slugs and files; compare tokenizer and data splits, PAD masks, model parameter count, output shapes, and finite forward/backward; verify that GloVe-related code is absent from #4; verify `requirements.py`, source notebook and model exporter align. After running, record **actual run ID, version, validation metrics, status, threshold, best epoch, seed, model size, environment and downloaded ZIP** in a future handoff update. Do not declare #4 completed merely because its notebook exists.

**For subsequent core IDs** preserve the same family selection rule: #5 GloVe joint additive → derive from latest non-ablation GloVe architecture; #6 WordPiece joint additive → latest WordPiece; #7/#8 corresponding cross dot-product; #9/#10 joint dot-product. The new architecture is a *focused change* to the matching family notebook plus appropriate naming/artifact updates. For each, choose whether residuals, pooling, projection, packing, and other surrounding blocks are matched before comparing results; never silently add a full Transformer block just because attention is dot-product.

### 5. End-of-session validation checklist for each new experiment

- [ ] Match new core ID and **tokenizer family** to the correct latest baseline notebook; do not cross-wire GloVe and WordPiece assets.
- [ ] Preserve model/trainer/MLflow/classes/parameters unless the agreed architecture requires a specific change; avoid unsolicited extra cells, migration frameworks or synthetic notebook outputs.
- [ ] New **canonical** notebook exists in dataset at the exact path used by setup, `NOTEBOOK_PATH`, model exporter and requirements extraction; dataset is versioned **before** the run.
- [ ] Set unique `ATTENTION_TYPE`, output directory, filename, architecture export and MLflow run name; preserve historical runs.
- [ ] Check Kaggle source files, CUDA/PyTorch support (historical P100 needed `sm_60` compatible build), paths, package versions and MLflow schema compatibility.
- [ ] Check train/val provenance, vocabulary or trained tokenizer, all masks, empty/all-masked behavior, packing and directional cross-/joint-attention; run forward/backward tests with padding and unequal question lengths.
- [ ] Verify GloVe vector loading/coverage and freeze schedule **only for GloVe runs**; WordPiece uses fresh trainable embeddings.
- [ ] Confirm `model_architecture.py` equivalent can import with all required dependencies/config globals; checkpoint loads and model predictions match. For #3B verify `pooling_keep_mask` buffer.
- [ ] Confirm final ZIP explicitly contains the **unique architecture `.py` at ZIP root**, current SQLite backup, MLflow artifacts and correct experiment-specific model output files.
- [ ] Confirm final cell actually restored original Windows artifact URIs; retain backup and extract ZIP to the matching project layout. Verify DB schema/version prior to opening Windows UI.
- [ ] For true executed notebook outputs use a saved Kaggle notebook version; do **not** equate dataset source notebook with a live executed copy, or overwrite historical outputs without checking run ID.
- [ ] Update status with measured run IDs/results **only when supplied or verified**. Keep historical September entries intact rather than conflating with new experiments.

---

## Full QQP research roadmap — agreed 2026-10-15

**Scope:** This five-stage roadmap is the user's intended order for the overall Quora Question Pairs (QQP) research project. It is broader than the original September 2026 handoff, which concentrated on BiLSTM–attention architecture experiments. The roadmap is a plan, **not evidence that the later stages have been implemented or trained**.

1. **LSTM–Attention experiments (GloVe and WordPiece).** Compare the **10 numbered architectures in the “Core experiment matrix”** below: word-level + pretrained GloVe versus the trained custom WordPiece tokenizer + learned embeddings, each across independent attention pooling, additive cross-attention, additive joint attention, PyTorch dot-product cross-attention, and PyTorch dot-product joint attention. Keep the existing BiLSTM backbone and trainer while screening. Treat experiment IDs (#1–#10) as distinct from MLflow *run version* labels (v1, v2, v10, etc.).
2. **ESIM experiments.** Develop and evaluate ESIM-style question-pair models as a separate model-family phase, informed by the historical ESIM-inspired notebook. Clearly distinguish an ESIM-inspired adaptation from a faithful reproduction of the original ESIM architecture; the historical ESIM run suffered the GloVe loading bug described below and is not a reliable pretrained-GloVe benchmark.
3. **Custom BERT-style Transformer encoder from scratch.** Reuse the project's **custom-trained WordPiece tokenizer** with newly initialized, trainable embeddings and Transformer encoder weights to train a question-pair classifier. Specify pair-formatting, question/segment embeddings, position embeddings, bidirectional attention, feed-forward layers, residuals, and normalization. **Training a BERT-style encoder for QQP classification from scratch is not equivalent to pretraining BERT with masked-language modeling**; an MLM pretraining phase is not assumed unless separately decided. This stage is separate from the BiLSTM + joint-attention experiments.
4. **Fine-tune a pretrained BERT from Hugging Face.** Select a pretrained BERT checkpoint and fine-tune it for the QQP classification task. Use the checkpoint's **matching pretrained tokenizer by default**; do not substitute the custom WordPiece vocabulary while assuming pretrained input embeddings still match. Preserve checkpoint/tokenizer provenance and log training settings.
5. **Compare the model families and results.** Compare tuned LSTM–Attention finalists, ESIM finalists, the from-scratch Transformer, and fine-tuned pretrained BERT. Include held-out classification quality (particularly F1), threshold policy, training/inference costs, parameter count, and reproducibility. Distinguish results selected on validation data from final evaluation on untouched test data.

### Architecture screening and hyperparameter optimization strategy

**Agreed workflow for Stage 1:** **screen all 10 architectures → shortlist ~2–4 promising candidates → tune each shortlisted candidate → compare best settings across multiple fixed seeds → select the winner → evaluate once on an untouched test set.** The shortlist size is a practical guideline rather than an inflexible quota; retain close performers and efficient alternatives when justified. A leading architecture under a fixed hyperparameter configuration is not necessarily the winner after tuning.

- **Screening:** Keep train/validation split, label definition, model-selection metric, threshold procedure, reasonable shared baseline hyperparameters, and evaluation/reporting protocol comparable. Use meaningful architecture-specific settings where necessary; log parameter count, sequence lengths, and compute budget. WordPiece and word-level GloVe vary in both tokenization **and** pretrained initialization, so do not attribute their difference to attention alone.
- **Shortlisting:** Consider validation F1, uncertainty across runs, stability, speed, memory, and parameter count; avoid eliminating a near-tied model solely because its first-run F1 is marginally lower.
- **Tuning (HPO):** Allocate comparable trial budgets across shortlisted architectures. Initially optimize learning rate, weight decay, dropout, and classifier settings; consider model width, heads, depth, and other architecture-specific choices later. Keep validation tuning separate from final test evaluation.
- **Finalists and test:** Repeat promising configurations across multiple fixed random seeds (e.g., 3–5 if affordable); compare distributions rather than a single best run. Reserve an untouched test split for the final selected model or a clearly defined final comparison protocol.
- **Later stages:** Apply the same *principle* of controlled screening, selective HPO, and final evaluation to ESIM and Transformer families, but do not force the exact same search space or learning rate on unlike architectures. A pretrained BERT fine-tuning recipe and a randomly initialized custom Transformer need different sensible training ranges and budgets.

**Status boundary:** The original run inventory and verification in this document reflect the historical September 2026 audit. Any later Kaggle/local training reports or newly completed variants should be recorded with exact run IDs, source notebooks, and metrics before revising historical status claims. **No new training is started by this roadmap update.**

## Files and environment to inspect first

Project root: `C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs`.

- [Current project training notebook](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/02-LSTM-attention-train.ipynb).
- [User-supplied Kaggle notebook reviewed in this conversation](<C:/Users/98922/Downloads/02-lstm-attention-train (1).ipynb>). This may differ from the repository copy. Confirm which is the latest before editing.
- [Final WordPiece tokenizer notebook](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/01-train-wordpiece-tokenizer-final.ipynb).
- [Other WordPiece tokenizer notebook](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/01-wordpiece-tokenizer-train.ipynb).
- MLflow tracking database: `C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlflow-quora-questions-pairs.db`.
- Historical artifacts: `C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/<run_id>/artifacts/`.

For historical architecture review, inspect the saved **training notebook** first, as the user requested. Summaries and the tracking database provide results and status. Exported architecture files can become stale if their notebook source was stale.

The established Kaggle input included:

`/kaggle/input/datasets/amirmohamadaskari/quora-question-pairs-wordpiece-training-assets/data/processed/train_split.csv`

The user chose copying the required project assets to a writable working directory so existing relative paths could continue working. Preserve that choice unless the next session changes it. Verify actual paths rather than assuming a mount or working directory.

The reviewed Kaggle notebook had a working P100 setup with PyTorch `2.8.0+cu126`. A different installed build had detected the GPU but lacked `sm_60` support. Recheck the actual runtime before training; do not upgrade solely to match documentation links in this document.

## Definitions settled in the conversation

### Tokenization and initialization are separate choices

The user's shorthand is interpreted as:

1. **GloVe setup:** word-level tokenizer and pretrained GloVe word embeddings, with an explicit policy for unmatched words and freezing/unfreezing.
2. **WordPiece setup:** the user's custom WordPiece tokenizer and randomly initialized, learned token embeddings. This does not imply pretrained BERT weights.

GloVe is an embedding source; WordPiece is a tokenizer. Standard GloVe word vectors do not directly provide a complete embedding table for the WordPiece vocabulary. Comparing these two setups changes both segmentation and initialization, so it does not isolate the effect of tokenization alone.

### Independent attention pooling

Each question goes through the shared BiLSTM and is pooled independently. The classifier receives the two pooled representations and comparison features. The original W/v attention is learned additive pooling over contextual BiLSTM outputs; it is not token-to-token self-attention merely because it processes one sentence.

Its limitation is the timing of interaction: compression happens before the question can inspect its partner. This is not evidence that LSTMs are unsuitable or that cross-attention must perform better.

### Explicit cross-attention in both directions

Within one model call `model(q1, q2)`:

- Q1 queries the keys/values from Q2, producing one output per Q1 token.
- Q2 queries the keys/values from Q1, producing one output per Q2 token.

These are two attention operations inside one model forward pass. A single direction is a valid architectural option, but it does not automatically return representations for the opposite direction. Using both directions also does not guarantee identical classifier predictions after swapping the questions.

### Combined-sequence attention

Concatenate the question token representations and apply unrestricted self-attention over the combined sequence. Every valid token can attend to both its own question and the other question. The mechanism is self-attention; the pair model performs joint interaction and can be described as a cross-encoder/hybrid interaction model.

For the first controlled comparison, the recommendation is:

```text
Q1 -> shared BiLSTM --+
                     +-> concatenate token outputs -> joint attention
Q2 -> shared BiLSTM --+
```

Then split outputs back by question, apply the same pooling, and retain the same pair-classification head. This keeps the interaction pattern as the main change. Concatenating the inputs **before** the BiLSTM is a separate optional experiment because the recurrent encoding then also crosses the question boundary.

“Transformer style” needs a precise definition. Joint attention alone is not a complete Transformer encoder block. A complete block also includes a position-wise feed-forward network, residual connections, and normalization. Decide and document these components before implementation; do not silently add them only to the PyTorch variants.

Segment IDs 0 and 1 select two learned vectors that identify the question. They are not scalar values directly added to token embeddings. Their use, position embeddings, separators, and any CLS token are explicit design choices. The existing BiLSTM already supplies order-sensitive context; adding positional embeddings after it is an additional choice. A Transformer-only replacement is outside the initial LSTM experiment scope.

### Additive versus dot-product scoring

The proposed “custom” attention means the user's additive/Bahdanau token-pair attention:

`score(i,j) = v_score^T tanh(W_q h_i + W_k h_j)`

Attention weights are normalized over valid source tokens and applied to their value representations. The small scoring vector `v_score` is distinct from the value vectors. Q/K/V are conceptual roles; an API can combine keys and values in one `key_value` argument.

The proposed PyTorch attention means `torch.nn.MultiheadAttention`, using scaled dot-product scoring. Thus “custom versus PyTorch” compares scoring mechanisms as well as implementation. It is not a pure library benchmark.

An earlier assistant suggestion added PyTorch `self.cross_attention` before the user's existing `self.attention` pooling layer. It did **not** turn the existing pooling class into cross-attention by passing three inputs to it. The saved historical models instead implement custom additive cross-attention directly. Both approaches are valid.

## Core experiment matrix

The user explicitly requested four interaction designs for each embedding/tokenizer setup. Two original pooling baselines were recommended, producing this ten-experiment matrix. Preserve these experiment numbers in future discussion.

| ID | Tokenization / embeddings | Attention after BiLSTM | Historical status |
|---|---|---|---|
| 1 | Word-level + GloVe | Independent additive attention pooling | Already tried; MultiHead-Bahdanau v9/v10 have verified corrected GloVe loading |
| 2 | WordPiece + learned embeddings | Independent additive attention pooling | Current notebook family; reviewed Kaggle snapshot showed training in progress, not verified final completion |
| 3 | Word-level + GloVe | Custom additive cross-attention in both directions | Architecture already tried in CrossAttention-Bahdanau v1–v6; those runs have the GloVe lookup bug and need corrected runs |
| 4 | WordPiece + learned embeddings | Custom additive cross-attention in both directions | Proposed; no completed matching run established by this audit |
| 5 | Word-level + GloVe | Custom additive attention over combined sequence | Proposed |
| 6 | WordPiece + learned embeddings | Custom additive attention over combined sequence | Proposed |
| 7 | Word-level + GloVe | PyTorch dot-product cross-attention in both directions | Proposed |
| 8 | WordPiece + learned embeddings | PyTorch dot-product cross-attention in both directions | Code was previously suggested in conversation; no completed matching run established |
| 9 | Word-level + GloVe | PyTorch dot-product attention over combined sequence | Proposed |
| 10 | WordPiece + learned embeddings | PyTorch dot-product attention over combined sequence | Proposed |

“Proposed” means this conversation did not establish a completed matching run, not that no such experiment could exist elsewhere. The audit searched completed runs and their saved notebooks in the LSTM_attention artifact family.

## Additional options discussed

These are optional follow-ups, not an instruction to expand every combination immediately:

**New consideration — dedicated within-question token-to-token self-attention ablation (not part of the 10 core experiments):** After the shared BiLSTM, let Q1 tokens attend only to Q1 tokens and Q2 tokens attend only to Q2 tokens, without cross-question interaction before final pair comparison. Consider both custom additive/Bahdanau scoring and PyTorch scaled dot-product multihead scoring, with matched downstream pooling and classifier. This differs from experiments #1–#2 (independent additive *pooling*, not token-to-token self-attention) and from #5–#6/#9–#10 (joint token-to-token attention over both questions, including cross-question interactions). The research question is whether any improvement comes from richer **within-question token contextualization** or from **direct inter-question interaction**. Prioritize completing and screening the original 10 experiments first; revisit this as a targeted ablation on selected candidates only if results justify it, **not** as an automatic expansion of the core matrix.

- **Attention masking ablation for Experiment #3 (not a new core experiment ID):** Compare **(#3A) PAD-only masking** (current implementation; all real tokens, including stopwords, remain eligible for attention) against **(#3B) PAD + stopword masking in final attention pooling only** (stopword token representations receive no direct pooling weight while remaining available to the BiLSTM and cross-attention). Keep the same trained architecture, tokenization, dataset split, hyperparameters, and evaluation/threshold procedure across variants; only the final-pooling mask policy should differ in this first controlled comparison. Record the exact stopword list and handling of punctuation/UNK, and make all-masked pooling safe. Masking stopwords in **cross-attention keys/values** is a *different, separate optional ablation* because it restricts inter-question alignment; do not silently combine the two changes. Preserve the existing Experiment #1 stopword-masked baseline; if needed, compare against a separate PAD-only/packed-BiLSTM #1 rerun to isolate architectural effects. This tests whether retaining potentially important function words and negations improves QQP prediction, without assuming either mask wins.

- **Pooling ablation after cross-attention (especially Experiment #3; not a new core experiment ID):** Compare three *alternative* ways to summarize the cross-attention-enriched token sequences before the same pair classifier: **(A) multihead additive attention pooling** (current Experiment #3), **(B) masked mean pooling**, and **(C) masked mean + masked max pooling**. Each reduces variable-length token representations to fixed-size per-question features; none passes the full sequence directly into the classifier. Keep the BiLSTM, bidirectional cross-attention, padding policy, dataset split, evaluation/threshold procedure, and training setup matched. Mask PAD in all pooling variants; define safe handling for empty/all-masked sequences. Mean+max concatenation changes feature width, so adapt the classification head and report parameter counts to avoid mistaking added capacity for a pooling benefit. Test whether learned attention pooling actually improves QQP classification compared with simpler summaries; treat this as a *targeted ablation* after initial architecture screening, not a replacement of the 10-experiment matrix.
- **Self-attention followed by cross-attention:** first update tokens within each question, then exchange information between questions. A historical Self+Cross run exists, but shares the GloVe loading bug.
- **ESIM-style composition:** cross-attention, local difference/product features, another BiLSTM, then pooling. A historical ESIM-style run exists, also affected by the GloVe bug.
- **Word-level random embeddings:** a useful control for separating pretraining effects from tokenizer effects. Do not automatically treat the buggy old runs as this controlled baseline: they also froze random embeddings initially and differed in other settings.
- **Joint input before BiLSTM:** concatenate questions before recurrent encoding instead of concatenating independent BiLSTM outputs.
- **One-direction cross-attention:** tests whether both directions are necessary for a chosen classifier; account for question-order dependence.
- **GloVe freeze policy, pooling method, and attention depth:** vary individually after initial comparisons, rather than changing all simultaneously.
- CNN placement was explored historically (cross v4–v6). Keep those as references; CNN expansion is not part of the initial matrix.

## Verified historical findings and corrections

### Cross-attention computation is real and correctly wired

Eight completed cross-attention notebooks were inspected: CrossAttention-Bahdanau v1–v6, Self+Cross, and ESIM-style. All eight use the same additive `CrossAttentionHead` computation. Both directions use the correct opposite-question key/value sequence and key mask, normalize over source tokens, and aggregate source values. Head concatenation, output projection, residual addition, and normalization have consistent dimensions in the inspected designs.

Small CPU checks using the exact saved head code confirmed correct output shape for unequal query/key lengths, dependence on valid source tokens, and invariance to changes in masked source tokens when at least one source token is valid. These were focused attention checks, not a reproduction of full training.

### All-masked attention edge case — all eight cross runs

When every source token is excluded, filling all scores with finite `-1e10` produces a uniform softmax, so the output averages excluded values. This was reproduced. Stopword and UNK masking makes this possible beyond empty input. Define behavior for empty/all-masked sequences and test it. Switching to `torch.finfo(dtype).min` alone does not solve the all-masked case; replacing with negative infinity can instead produce NaNs if all scores are excluded.

Use dtype-safe masking for mixed precision as a separate concern. `-1e10` cannot be represented as a finite float16 number.

### GloVe lookup bug — all eight cross runs

The saved notebooks build `idx2word` as integer ID -> word, but their loader checks string words against that dictionary's keys. No pretrained vectors are copied. The returned matrix remains randomly initialized, despite successful-looking loading messages and GloVe metadata. Embeddings are initially frozen under those configurations and subsequently unfrozen by the training schedule.

Use the word -> ID mapping (`tokenizer.vocabs`) for GloVe lookup. Verify actual loaded-vector count and sample values, dimensions, PAD handling, and unmatched-word initialization.

**Do not generalize this bug to all baseline runs.** Independent-pooling baseline v9 and v10 already have a corrected loader using `self.vocabs`. Their saved notebook outputs both report **18,604 / 20,000 vectors, 93.02% coverage**. Their recorded validation F1 scores are **0.81552** and **0.82616**, respectively. They establish that experiment 1 was tried with actual GloVe loading. Experiment 3 was tried architecturally, but the inspected runs do not establish the intended GloVe-initialized result.

### Padding through recurrence — all eight cross runs

The inspected cross notebooks feed padded sequences through ordinary BiLSTMs without packing and construct embeddings without `padding_idx`. Masking attention/pooling afterward does not prevent the backward recurrent state from processing padding. Handle true sequence lengths through recurrent layers, including ESIM composition if used. A zero PAD vector alone does not remove recurrent padding steps. Handle CNN boundaries consistently in any later CNN experiment.

### Stopwords and question order

The older tokenizers exclude NLTK English stopwords, punctuation, PAD, and UNK from attention; this is not merely frequent-token analysis in those versions. Document the mask policy. For QQP, masking words such as negation deserves a controlled decision. Masked tokens can still affect contextual BiLSTM states, so do not claim they were removed entirely from the model.

Bidirectional cross-attention and shared weights do not enforce swap-invariant classification. Ordered concatenation of Q1/Q2 features can produce different predictions for `(q1,q2)` and `(q2,q1)`. Measure this and choose an explicit policy; do not label it automatically symmetric.

### Misleading flags and ESIM naming

Cross v4, v5, and v6 log `USE_SELF_ATTENTION=True`, but their forward methods do not invoke self-attention. The separate Self+Cross experiment invokes it correctly. Historical configuration flags for residuals also do not always control the hardcoded forward behavior; inspect execution paths before reusing flags.

The ESIM-style version has alignment, enhanced local features, composition BiLSTM, and average/max pooling. Its additive multihead attention and residual/normalization changes make it an ESIM-inspired variant, not an exact original-paper reproduction.

### Training reports and source capture

All eight cross runs have valid `run_sammary.json` files (the filename is misspelled exactly this way). Their `training_history.json` files contain an object-description string rather than the actual dictionary. Epoch metrics remain available in the MLflow SQLite database. Log the underlying history dictionary; do not serialize the `TrainingHistory` object itself.

The current Kaggle notebook was previously reviewed as logging the history dictionary every epoch; do not reintroduce the older bug or assume every version has it.

The export/requirements helpers read notebook JSON at `NOTEBOOK_PATH`, not the live notebook's cells. Keep that source synchronized with the code actually trained. A stale dataset notebook can cause stale architecture exports and incorrect dependency detection. Preserve automatic generation and logging, while verifying its source. Do not remove those features to avoid fixing the path/source issue.

## Fair comparison protocol — recommendations for agreement

1. Use the same stored train/validation split and label meaning. Inspect preprocessing and tokenizer provenance; fit tokenizers/vocabularies on training data only. Evaluate a held-out test set after model selection when available.
2. Within each tokenizer/embedding setup, hold BiLSTM dimensions, pooling, classifier, loss, scheduler, stopping/selection rules, and training budget consistent where possible. Record parameter-count differences; additive and dot-product heads are not automatically equal in capacity.
3. Keep residual, normalization, feed-forward, segment/position handling, and attention depth explicit. Prefer matched surrounding blocks so attention comparisons do not accidentally compare many architectural changes.
4. Different tokenizers produce different sequence lengths. Report truncation rates and question coverage; equal token limits do not necessarily retain equal text. Log the maximum lengths and effective batch sizes.
5. Use validation to choose thresholds and checkpoints. Report F1, accuracy, precision, recall, AUROC, average precision, threshold, best epoch, and completed epochs. Distinguish validation-selected results from independent test results.
6. Record runtime, peak GPU memory, trainable/total parameters, seed, tokenizer identity, GloVe coverage and freeze policy, and exact runtime versions. Repeat promising comparisons with multiple fixed seeds before treating small differences as reliable.
7. Log the actual architecture choice and notebook source for every run; save outputs per experiment. Never delete tokenizers or prior checkpoints as incidental setup cleanup.

## Suggested implementation order

1. Inspect the latest notebook and select the starting version with the user. Compare it with the successful v9/v10 baseline notebooks and the simplest cross v1 notebook.
2. Resolve common correctness issues in the selected starting code before comparing architectures. Keep the user's existing trainer and automation.
3. Establish or verify baselines 1 and 2 under the chosen protocol; existing historical metrics are references and may not satisfy a new matched protocol.
4. Implement the simplest custom cross variant and its masks, preserving `model(q1, q2)` -> one logit per pair where practical. Test before a long run.
5. Implement the corresponding dot-product variant with the same surrounding design.
6. Implement combined-sequence variants with explicit boundaries and matched pooling/classifier. Confirm whether the user wants only joint attention or full Transformer-style blocks.
7. Run the eight interaction experiments in an agreed order and compare them with the baselines. Select optional extensions from evidence rather than launching every permutation.

## Correctness checks before long training

- Tensor/output dimensions and a finite forward/backward step for unequal question lengths.
- Valid partner-token changes affect pre-pooling representations; excluded key/value changes do not affect the attention head when at least one key is valid.
- Empty/all-masked rows have defined finite behavior, and PAD/UNK/stopword policies match the chosen design.
- Full-model padding/length handling is correct, beyond the attention-head mask alone.
- For joint attention, both question masks and boundaries remain correct after concatenation/splitting; attention is bidirectional, not accidentally causal.
- Swapping question order has measured behavior; symmetry is not assumed.
- GloVe vectors actually load; PAD handling and freeze/unfreeze behavior are verified.
- Exported architecture and logged notebook correspond to the trained model; history JSON contains metrics; run configuration reflects executed modules.

## Reference documentation

- [GloVe word vectors](https://nlp.stanford.edu/projects/glove/).
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
- [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
- [Original ESIM paper](https://aclanthology.org/P17-1152.pdf).

Historical results below describe saved runs. They do not demonstrate that additive attention, dot-product attention, cross-attention, or independent pooling is universally superior.

## Historical run inventory and reports

The tracking database contained 20 FINISHED runs in this inspected artifact family. Links open the original saved training notebooks and full summary reports. Epochs are shown as best validation-F1 epoch / total recorded validation-F1 epochs, numbered from 1. Metrics are recorded validation results, not newly reproduced test scores.

| Run / training notebook | Report | Best / total epochs | F1 | Accuracy | AUROC | Minutes |
|---|---|---:|---:|---:|---:|---:|
| [LSTM_attention-OneHead-Bahdanau-v1](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/2d484d08a85248e693e67cdf74493652/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/2d484d08a85248e693e67cdf74493652/artifacts/run_sammary.json) | 6 / 13 | 0.76769 | 0.81753 | 0.88812 | 99.15 |
| [LSTM_attention-OneHead-Bahdanau-v6](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/89a817511b734e1abd735c4f5e3d4050/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/89a817511b734e1abd735c4f5e3d4050/artifacts/run_sammary.json) | 7 / 14 | 0.78093 | 0.82814 | 0.89605 | 147.36 |
| [LSTM_attention-MultiHead-Bahdanau-v1](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/f3572bb9feca4f348686372594fcb767/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/f3572bb9feca4f348686372594fcb767/artifacts/run_sammary.json) | 7 / 9 | 0.77613 | 0.82898 | 0.89530 | 146.63 |
| [LSTM_attention-MultiHead-Bahdanau-v2](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/faef31782a0f4b02b6448aab392a24e5/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/faef31782a0f4b02b6448aab392a24e5/artifacts/run_sammary.json) | 9 / 9 | 0.77527 | 0.82812 | 0.89421 | 219.01 |
| [LSTM_attention-MultiHead-Bahdanau-v3](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/08743c6873cf492682be47888737f248/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/08743c6873cf492682be47888737f248/artifacts/run_sammary.json) | 9 / 12 | 0.77342 | 0.82703 | 0.89083 | 291.88 |
| [LSTM_attention-MultiHead-Bahdanau-v4](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/591e35cbb334420db7303a53d435009c/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/591e35cbb334420db7303a53d435009c/artifacts/run_sammary.json) | 6 / 6 | 0.79996 | 0.84719 | 0.91773 | 48.97 |
| [LSTM_attention-MultiHead-Bahdanau-v5](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/868c3da1a9a844ceb8021c71c17c0147/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/868c3da1a9a844ceb8021c71c17c0147/artifacts/run_sammary.json) | 8 / 8 | 0.80524 | 0.85206 | 0.91943 | 129.37 |
| [LSTM_attention-MultiHead-Bahdanau-v6](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/c3f2f9597fb94be29d8068e86166a519/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/c3f2f9597fb94be29d8068e86166a519/artifacts/run_sammary.json) | 9 / 9 | 0.81312 | 0.85656 | 0.92344 | 147.02 |
| [LSTM_attention-MultiHead-Bahdanau-v7](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/3fa2430b031340148fc73b00128aa4ba/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/3fa2430b031340148fc73b00128aa4ba/artifacts/run_sammary.json) | 10 / 10 | 0.81731 | 0.86136 | 0.90593 | 245.60 |
| [LSTM_attention-MultiHead-Bahdanau-v8](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/8e6697103ec04066aa25bfde4a4431e9/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/8e6697103ec04066aa25bfde4a4431e9/artifacts/run_sammary.json) | 10 / 10 | 0.80641 | 0.85174 | 0.91649 | 261.83 |
| [LSTM_attention-MultiHead-CrossAttention-Bahdanau-v1](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/1fdc7ccd9f854ccb961a593faeb7bd96/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/1fdc7ccd9f854ccb961a593faeb7bd96/artifacts/run_sammary.json) | 9 / 11 | 0.81238 | 0.85219 | 0.92369 | 166.68 |
| [LSTM_attention-MultiHead-CrossAttention-Bahdanau-v2](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/c14abfbd8c044f1299385b7b6982aacb/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/c14abfbd8c044f1299385b7b6982aacb/artifacts/run_sammary.json) | 7 / 8 | 0.81061 | 0.85221 | 0.92445 | 229.62 |
| [LSTM_attention-MultiHead-CrossAttention-Bahdanau-v3](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/f8eea01e92544e91bd5cd5e34fca6f3c/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/f8eea01e92544e91bd5cd5e34fca6f3c/artifacts/run_sammary.json) | 12 / 12 | 0.81156 | 0.85201 | 0.91488 | 196.76 |
| [LSTM_attention-Self+CrossAttn-MultiHead-v1](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/dab57bae733545aeb256c1b5e66bfea8/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/dab57bae733545aeb256c1b5e66bfea8/artifacts/run_sammary.json) | 10 / 11 | 0.80876 | 0.85219 | 0.92043 | 294.90 |
| [LSTM_attention-MultiHead-CrossAttention-Bahdanau-v4](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/0f769eba629540708924aee38d304101/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/0f769eba629540708924aee38d304101/artifacts/run_sammary.json) | 10 / 10 | 0.80611 | 0.85095 | 0.91969 | 166.34 |
| [LSTM_attention-MultiHead-CrossAttention-Bahdanau-v5](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/5ddfb6d332fc402a915ae39ec54aa4d1/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/5ddfb6d332fc402a915ae39ec54aa4d1/artifacts/run_sammary.json) | 11 / 12 | 0.79426 | 0.84291 | 0.90758 | 250.15 |
| [LSTM_attention-MultiHead-CrossAttention-Bahdanau-v6](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/7790cb095fe0455d9d9011b9187b2114/artifacts/LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/7790cb095fe0455d9d9011b9187b2114/artifacts/run_sammary.json) | 13 / 13 | 0.79663 | 0.84167 | 0.91552 | 202.55 |
| [LSTM_attention-ESIM_Style-MultiHead-CrossAttention-Bahdanau-v1](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/5bfde7c7f40d4620aa9c05a19de695cb/artifacts/LSTM-attention-train-ESIM.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/5bfde7c7f40d4620aa9c05a19de695cb/artifacts/run_sammary.json) | 10 / 10 | 0.79235 | 0.83690 | 0.91421 | 191.00 |
| [LSTM_attention-MultiHead-Bahdanau-v9](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/43315c63c4324b7eab42124068ce4e3a/artifacts/02-LSTM-attention-train-1.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/43315c63c4324b7eab42124068ce4e3a/artifacts/run_sammary.json) | 8 / 8 | 0.81552 | 0.85790 | 0.92992 | 129.54 |
| [LSTM_attention-MultiHead-Bahdanau-v10](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/ef0dd36d4f7745ae996f1e8709bf7a92/artifacts/02-LSTM-attention-train.ipynb) | [JSON](C:/Users/98922/Documents/python_scripts/AI/projects/quora-questions-pairs/mlruns/Quora-Question-Pairs/LSTM_attention/ef0dd36d4f7745ae996f1e8709bf7a92/artifacts/run_sammary.json) | 14 / 15 | 0.82616 | 0.86490 | 0.92724 | 240.73 |

### Mapping the eight cross runs to their actual variants

| Short name | Run ID | Actual variation |
|---|---|---|
| Cross v1 | `1fdc7ccd9f854ccb961a593faeb7bd96` | BiLSTM hidden size 256; cross-attention and masked mean pooling |
| Cross v2 | `c14abfbd8c044f1299385b7b6982aacb` | BiLSTM hidden size 384; cross-attention and masked mean pooling |
| Cross v3 | `f8eea01e92544e91bd5cd5e34fca6f3c` | Cross-attention followed by attentive pooling |
| Self+Cross | `dab57bae733545aeb256c1b5e66bfea8` | Token-to-token self-attention within each question, then cross-attention |
| Cross v4 | `0f769eba629540708924aee38d304101` | Single CNN kernel after BiLSTM, then cross-attention |
| Cross v5 | `5ddfb6d332fc402a915ae39ec54aa4d1` | Multiple CNN kernels after BiLSTM, then cross-attention |
| Cross v6 | `7790cb095fe0455d9d9011b9187b2114` | Multiple CNN kernels before BiLSTM, then cross-attention |
| ESIM-style | `5bfde7c7f40d4620aa9c05a19de695cb` | Cross-attention, enhanced local features, composition BiLSTM, average/max pooling |

All eight rows above are affected by the GloVe lookup bug. Their common recorded tokenizer settings were word-level vocabulary 20,000 and maximum length 50; they are not the current WordPiece setup. Cross v1 had the highest recorded F1 among these eight (0.81238), with precision 0.76439, recall 0.86681, and threshold 0.35040. These numbers do not establish superiority over the corrected GloVe baseline v10 (F1 0.82616), because training configurations and embedding initialization were not matched.

Independent-pooling baseline v9 (`43315c63c4324b7eab42124068ce4e3a`) and v10 (`ef0dd36d4f7745ae996f1e8709bf7a92`) both have corrected GloVe loading and saved 93.02% coverage output. Earlier baseline runs are included for discovery; do not assume their loader, loss, classifier, or pooling matches v9/v10. Some early models return a pairwise distance rather than the later classifier logit.
