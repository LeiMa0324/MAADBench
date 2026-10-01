MADBench — Anomaly Detection (benchmark_AD)
=========================================

This package evaluates anomaly detectors on the JSON traces produced by
main.runner / main.run_trace_matrix. Commands below run from the repository root.
Raw traces are read from output/traces; AD results are written to output/ad.
Existing trace files and historical plots/data are never overwritten.

QUICK START
-----------

0. Put your trace JSON files under output/traces/ at the REPOSITORY ROOT
   (MAADBenchCode/output/traces/), not inside benchmark_AD/.

   Both directory layouts below are supported, including a mixture of them:

   MAADBenchCode/
     output/
       traces/
         run_<timestamp>_<domain>_<tool|no_tool>_<model>_<temperature>/
           <trace_file>.json

   MAADBenchCode/
     output/
       traces/
         <model>/
           gsm-hard/                  # or livecodebench/
             run_<...>/
               <trace_file>.json

   The loader searches recursively. Use the original per-room trace JSONs
   produced by main.runner, containing config, room, trace and escaped;
   summary.csv, room definitions and prediction CSVs are not trace inputs.
   Existing traces generated into output/traces/ need no copying or moving.

   If your traces live elsewhere, pass their root directory when starting:

   bash benchmark_AD/run_methods.sh --path-traces "/absolute/path/to/traces"

   The default batch expects BOTH domains and BOTH tool/no_tool modes, at
   temperature 0. If you only have GSM-Hard tool traces, for example, use:

   bash benchmark_AD/run_methods.sh \
     --datasets EscapeRoom_v3_gsm_hard --trace-types tool

1. Create the one project environment (Python 3.12 required):

   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

   This is the only environment used by the benchmark runner, all registered
   AD methods, and Gemma. Do not create a separate AD or LLM environment.
   On NVIDIA CUDA, install the matching PyTorch wheel before this command.
   On macOS, XGBoost may also need `brew install libomp`. First use of an
   embedding method downloads sentence-transformers/all-MiniLM-L6-v2.

2. One command starts ALL registered AD methods:

   bash benchmark_AD/run_methods.sh

   Equivalent:

   python -m benchmark_AD.run_all

   Default matrix: 24 method entries x 2 domains x tool/no_tool x seed 1
   = 96 sequential jobs, temperature 0, all available backbone models.
   Methods run in separate subprocesses, one at a time to avoid GPU contention.
   A failed method does not stop later jobs. The batch exits nonzero if any job
   failed; inspect batch_manifest.json and that job's run.log/error.txt.
   Full training can take a long time. This is not a background service:
   keep the terminal/session alive, or run it in your scheduler/tmux session.

   Gemma exports the common train/test split and then runs inference with this
   same Python interpreter. Set HF_TOKEN in this environment if model access is
   gated. Large Gemma variants require sufficient accelerator memory.
   cnn, tspulse and gemma3_1b are not part of the runnable benchmark: their
   source tree refers to local modules that are absent from this repository.
   This is a missing-code issue, not an environment issue, so they are not
   included in "all".

3. Gemma inference is included by default. To run several Gemma variants:

   bash benchmark_AD/run_methods.sh \
     --llm-models gemma-4-E2B gemma-4-E4B gemma-4-31B gemma-4-26B-A4B

   Use `--skip-llm-inference` when you only want to export Gemma's shared data
   split. Model downloads happen normally through the same environment.

USEFUL STARTING COMMANDS
-----------------------

Preview the full job matrix without loading data/models:

   bash benchmark_AD/run_methods.sh --dry-run

Fast end-to-end data/output check, without embeddings or GPU:

   bash benchmark_AD/run_methods.sh --methods random

Run a smaller selection, one domain and tooling mode:

   bash benchmark_AD/run_methods.sh \
     --methods random random_forest isolation_forest \
     --datasets EscapeRoom_v3_gsm_hard --trace-types tool --seeds 1

Run all methods with five seeds:

   bash benchmark_AD/run_methods.sh --seeds 1 2 3 4 5

Limit backbone models / include additional temperatures:

   bash benchmark_AD/run_methods.sh \
     --models gpt-4.1 gpt-5.4 --temperatures 0 0.3 0.6

Override data and output roots (paths may contain spaces):

   bash benchmark_AD/run_methods.sh \
     --path-traces "/path/to/traces" --output-dir "/path/to/ad-results"

Validate loading/splits without training:

   python -m benchmark_AD.main --method random --prepare-only

One method (absolute default paths also work outside the package directory):

   python -m benchmark_AD.main \
     --method random_forest \
     --trace_dataset_name EscapeRoom_v3_gsm_hard \
     --trace_type tool --seed 1 --temp_llm 0

The old direct-script entry still works:

   python benchmark_AD/main.py --method random

Optional --epochs 1 (single and batch entry) shortens neural training for smoke
checks; it is not a substitute for the default training schedule. Classical
methods ignore this option. DeepSAD applies it to both pretraining and training.
Use PYTHON=/path/to/python bash benchmark_AD/run_methods.sh to select the AD
interpreter explicitly. Legacy batch_jobs/*.sh retain their historical cluster
paths; use run_methods.sh/run_all for the integrated project workflow.

WHERE ARE THE PREDICTIONS?
--------------------------

Every invocation creates a new timestamp/UUID directory; reruns do not append
predictions into an older experiment. The console prints the exact directory.

Batch layout:

   output/ad/
     batch_<UTC timestamp>_<unique id>/
       batch_manifest.json           commands, status, return codes, logs, timings
       batch_summary.csv             job statuses, final metrics, artifact paths
       EscapeRoom_v3_gsm_hard/
         tool/
           action/
             random_forest/
               seed_1/
                 config.json         effective CLI configuration and input root
                 status.json         running / completed / prepared / exported / failed
                 run.log             this subprocess's stdout + stderr
                 error.txt           traceback if the main run failed
                 train_manifest.csv  exact training split before method filtering
                 test_manifest.csv   exact test split
                 metrics.csv         per-epoch metrics (one row for classical ML)
                 predictions.csv     FINAL epoch predictions, one row per sample
                 model.pt            final PyTorch weights when supplied by method
                 devnet.keras        DevNet best training-loss checkpoint, if used
                 data/
                   train_features.csv  prepared tabular features (before semi-supervised selection)
                   test_features.csv
                   feature_columns.json
                   scaler.joblib
           trace/
             igad/seed_1/...
             rqgnn/seed_1/...
             ma2df_improved/seed_1/...
         no_tool/...
       EscapeRoom_v3_live_code_bench/...

Single runs have the same inner layout, under output/ad/<timestamp>_<unique id>/.
--run-dir can select an exact NEW run directory. An existing directory is rejected
so metrics/configuration from different runs cannot be accidentally mixed.
run.log is captured automatically by the batch launcher; a single run logs to
its terminal. Not every detector provides a saved model; predictions are the
common output contract. Tabular feature CSVs omit full raw JSON to avoid copying
large trace payloads into every feature row.

predictions.csv columns:

   test_id / row_id   stable sample identifier (same value, test_id kept for clarity)
   gt_label          0 normal, 1 anomalous
   pred_label        predicted binary label
   pred_score        anomaly score; larger means more anomalous
   source_file       path relative to config.json's path_traces
   action_id         original action ID (action mode)
   puzzle_id         original puzzle ID (action mode)
   step_num          zero-based execution position (action mode)
   trace_id          source trace identity (action mode)
   room_id or room_number, room_group, difficulty
   task, trace_type, model_name, temperature
   step_label or label  original label retained from the split manifest

Use (source_file, action_id, step_num) to locate an action in its original JSON.
To join a run's summary.csv / benchmark_analyze annotations, match the trace
file AND action_id, not action_id alone: the same room/action IDs can occur in
multiple models, temperatures and runs. Do not compare action metrics with
trace metrics as if they represented the same evaluation task.

The writer validates unique IDs, manifest membership and ground-truth alignment
before saving predictions. IDs depend on relative source path and action position,
not filesystem traversal order. Moving the entire trace root preserves IDs;
renaming internal paths intentionally changes them.

LLM JUDGE OUTPUT
----------------

The gemma4 export job produces:

   .../action/gemma4/seed_1/data/
     train.csv
     test.csv
     config.json

Each requested model writes its own predictions/metrics in the same environment:

   .../action/gemma4/seed_1/gemma-4-E2B/predictions.csv
   .../action/gemma4/seed_1/gemma-4-E2B/metrics.csv
   .../action/gemma4/seed_1/gemma-4-E2B.log

The export's status.json stays exported; batch_manifest.json records inference
status and per-model return codes. No enormous judge model checkpoint is copied.
To run inference manually in the activated project environment:

   python -m benchmark_AD.methods.llm_as_judge.run_gemma_4 \
     --data-dir "/absolute/path/to/gemma4/seed_1/data" \
     --method gemma-4-E2B --supervision zero_shot

The export's configuration supplies seed, dataset, tooling, level and threshold
policy. Both run_gemma_4.py and the compatibility entry run_gemma_4_full.py use
this SAME input schema. Few-shot mode uses training examples only.

INPUT DISCOVERY AND DATA PROCESSING
----------------------------------

main.runner produces output/traces/run_.../*.json. Archived runs may instead be
organized as output/traces/<model>/<domain>/run_.../*.json. Both are supported,
as is passing a single run directory. Files are discovered recursively and
sorted. Non-trace JSON manifests are ignored.

For v3, filters use JSON metadata rather than assuming folder positions:

   config.llm.model / temperature
   config.benchmark.clue_domain       gsm-hard or livecodebench
   config.benchmark.unit_tools        true or false
   room.room_id / difficulty
   trace[]
   escaped

Legacy gsm_hard folder spelling is accepted as gsm-hard. Folder-name fallback
is used only when domain/tool metadata is absent. --path_traces is respected;
it is no longer overwritten by --trace_dataset_name. The default model filter
is ALL models, including current Claude models. Empty filter results and malformed
trace records produce explicit errors. Invalid JSON files are not silently skipped.

Processing:

   JSON traces -> metadata/model/temperature filtering
               -> action rows OR whole-trace rows
               -> stratified split by (domain, full room ID, difficulty)
               -> method-specific encoding/training -> metrics + predictions

Both levels keep the same room group out of the opposite split across models,
temperatures and repeated runs. The group stratification label is whether any
sample in that room group is anomalous. A split requiring more rooms/classes
fails with an explanation rather than silently switching to an unsafe split.
Each method uses its level's deterministic split for a fixed corpus/filter/seed.
Action-level and trace-level splits can differ because their labels differ.
Within a level, the common train manifest is saved BEFORE OCC/semi-supervised
selection. Changing the corpus can change the split; archived manifests record
exactly what was used.

Labels:

   Trace:  escaped=true -> 0; escaped=false -> 1.
           Legacy filenames are a fallback only when escaped is missing.
   Action: verification.status=correct -> 0; wrong -> 1.
           Missing status plus explicit schema_errors -> 1.
           Other unknown/missing statuses default to an error. Choose
           --unknown-status anomaly for legacy non-correct-is-anomaly behavior,
           or --unknown-status drop to omit unlabelled actions explicitly.

An empty trace has no action predictions and is reported/omitted in action mode.
It remains in trace-level evaluation. Graph classifiers represent an empty trace
with one empty execution node, labelled from the trace outcome, without fabricating
an action ID. Whole-trace tabular statistics use missing-value sentinels.

Tabular action features:
   current input + accumulated agent/output messages through the current action
   -> cleaned text -> all-MiniLM-L6-v2 embedding
   + call duration/input tokens/output tokens (missing = -1)
   -> StandardScaler fit ONLY on training features -> detector.
   Ground-truth labels and room/model/source metadata are not model features.

Graph methods construct per-trace action chains with prompt/input/output encodings.
Sequence methods preserve execution order; the existing transformer window is
100 actions. Only those first 100 actions are predicted for a longer trace;
manifest joins preserve the corresponding row IDs (not one ID per trace).

OCC clean training keeps only normal prefixes before the first anomalous action;
trace-level OCC keeps normal traces. Polluted mode retains samples with normal
training labels. Semi-supervised methods use --perc_outliers_train, default 0.2.
Embedding extraction remains method-specific; there is no shared embedding cache
across jobs. A full matrix therefore repeats encoding work.

THRESHOLDS AND COMPARABILITY
---------------------------

The default --threshold-policy fixed does NOT consult test labels:

   Probability-producing classifiers: anomaly probability >= 0.5.
   Uncalibrated anomaly scores: top --anomaly-fraction of the test batch,
   default 0.1, with deterministic tie handling.

The latter is a transductive ranking budget, NOT a threshold calibrated on a
validation set or a known deployment prevalence. Choose the fraction explicitly
for your experiment. --threshold-policy oracle instead uses the number of true
test anomalies as top-k; it is provided for legacy-style research comparisons,
not deployment. Policies are written to config.json and metrics.csv. Historical
results may differ because splitting, IDs and default thresholds have changed.
AUC uses continuous scores and is not affected by binary threshold selection.

REGISTERED METHODS AND LEVELS
-----------------------------

Default action level:
   random, random_forest, svm, xgboost,
   knn, lof, isolation_forest, ocsvm, autoencoder, deepsvdd, devnet, deepsad,
   g_safeguard, g_safeguard_semi_supervised, blindguard, tam, dominant, ggad,
   anomaly_transformer, sup_transformer, gemma4.

Default / required trace level:
   igad, rqgnn, ma2df_improved.

anomaly_transformer, sup_transformer, ggad and gemma4 require action level.
Batch --dataset-level auto selects the appropriate level; forcing an incompatible
level produces an explicit error. Single --dataset_level follows the same rules.
Legacy EscapeRoom_v2 / gsm_hard loaders remain available with --path_traces;
legacy gsm_hard has trace-level labels only. The integrated default is v3.

VALIDATION
----------

   python -m pip install pytest
   python -m pytest benchmark_AD/tests -q

Pipeline tests check discovery/filtering, room isolation, stable IDs, unknown
statuses, output joins, batch failure isolation and LLM export. Optional method
integration tests use small synthetic traces and deterministic local embedding
stubs to avoid model downloads; they validate shapes and outputs, not detection
quality. Full accuracy/GPU/LLM evaluations require the actual models and runtime.
