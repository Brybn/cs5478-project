# Reproduction workflow

Every command runs from the repository root. `S` abbreviates the script runner:

```bash
S="uv run --frozen --no-sync python scripts"
```

**Conventions.**

- `ROBOTS_TASK` selects the task: `PickCube` (default), `StackCube` or `LiftPegUpright`. Export it for every
  command of the other two tasks.
- Every command that writes a dataset, checkpoint or result refuses to overwrite an existing output directory.
- Rollouts run on CPU, one thread per worker process. We used 8 to 12 workers on a 16-core machine.
  Policy training used one RTX 3070 (CUDA). Approximate times with 8 workers: nominal PickCube policy 1 h; each
  fine-tune 45 min with DDPM100 checkpoint selection (stages 1-3) or 10 min with DDIM10 (stages 4-5); each correction
  collection 1-25 min; PickCube monitor study and confirmation 35 min. The PickCube chain of Sections 3, 4 and 6
  takes about 8 h.
- Every rollout is determined by its reset seed, push (start step, duration, angle, force) and policy-noise seed.
  Noise for a policy call depends only on the noise seed and the absolute control step, so schedules that call
  the policy at the same step draw the same noise.
- New reset seeds exclude every seed already recorded in `data/`, `checkpoints/` and `results/`. The original project
  also kept `data/reset_seed_registry.json` (not in git), listing every seed used before its final cleanup; it is read
  when present and is not needed for a fresh run.
- `data/`, `checkpoints/` and `results/` are not in git. The paths below are those used for the report. The
  result files behind the report are exported to the tracked `report_results/` (Sections 8-9).
- Time to goal is simulated time (20 Hz) to the tenth consecutive success step. Episodes that never reach the goal
  are charged the full horizon.

## 1. Setup

```bash
uv sync --frozen --python 3.10
sudo apt-get install libvulkan1 mesa-vulkan-drivers     # CPU rendering on a fresh Ubuntu/WSL install
```

ManiSkill's motion planner (LiftPegUpright demonstrations and corrections) needs `mplib==0.1.1`, which crashes with
numpy 2. It therefore runs in a separate, git-ignored environment:

```bash
uv venv .venv-mp --python 3.10
uv pip install --python .venv-mp torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-mp "mani_skill==3.0.1" "mplib==0.1.1" "numpy<2" "sapien==3.0.3"
```

## 2. Demonstrations

PickCube: the official motion-planner demonstrations, replayed to state observations and end-effector delta
position control, then split 80/20 (split seed 4278).

```bash
uv run --frozen --no-sync python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos
$S/demos/replay_demos.py --traj-path data/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj -b physx_cpu --count 100
$S/demos/prepare_demos.py --source data/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
# -> data/split/{train,validation}.h5 (80/20 demonstrations)
```

The replay rebuilds the scene every episode (`reconfiguration_freq=1`). The split used for the report was built
from an earlier replay that reused scenes and corrected the stale grasp flags afterwards. A fresh run therefore
differs from it by less than 4e-4 in actions and 3e-3 in observations. All 80 training demonstrations succeed in
both. The replay itself is deterministic.

StackCube: all 1,000 official motion-planner demonstrations. 990 replay successfully; the split is 890/100.
Our copy of the download is stored as `data/stackcube_full/trajectory.{h5,json}`.

```bash
uv run --frozen --no-sync python -m mani_skill.utils.download_demo StackCube-v1 -o data/stackcube_full
$S/demos/replay_demos.py --traj-path data/stackcube_full/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj -b physx_cpu --count 1000
ROBOTS_TASK=StackCube $S/demos/prepare_demos.py \
  --source data/stackcube_full/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --output-dir data/stackcube/split --count 990 --train-count 890
```

LiftPegUpright: no official demonstrations exist, so we generate them with ManiSkill's planner (1,000 successes
from planner seeds 0-1139). Standing the peg up needs wrist rotation, so this task uses `pd_ee_delta_pose`
(7-D actions). Demonstrations longer than 250 steps (28) are dropped.

```bash
.venv-mp/bin/python scripts/demos/generate_demos.py -e LiftPegUpright-v1 -n 1000 --only-count-success -b cpu \
  --num-procs 1 --record-dir data/liftpeg_full --traj-name trajectory
$S/demos/replay_demos.py --traj-path data/liftpeg_full/LiftPegUpright-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o state --save-traj -b physx_cpu --count 1000
ROBOTS_TASK=LiftPegUpright $S/demos/prepare_demos.py \
  --source data/liftpeg_full/LiftPegUpright-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --output-dir data/liftpeg/split --count 970 --train-count 870 --max-steps 250
```

## 3. Nominal diffusion policies

Two observed states, 16 predicted actions, at most 15 executed per call. Training uses 100 DDPM noise levels.
The selected checkpoint (`best.pt`) samples with 100-step DDPM. All monitor studies use 10-step DDIM inference
copies with the same weights (`scripts/policy/ddim_copy.py`).

```bash
$S/policy/train_policy.py                       # PickCube, seed 1, 30k updates -> checkpoints/pickcube/stage0_nominal
$S/policy/train_policy.py --seed 2 --output-dir checkpoints/pickcube/seed2_nominal
ROBOTS_TASK=StackCube $S/policy/train_policy.py --demo-path data/stackcube/split/train.h5 \
  --output-dir checkpoints/stackcube/policy_120k --total-iters 120000 --eval-freq 10000 --eval-horizon 150 --num-eval-episodes 20
ROBOTS_TASK=LiftPegUpright $S/policy/train_policy.py --demo-path data/liftpeg/split/train.h5 \
  --output-dir checkpoints/liftpeg/policy_120k --total-iters 120000 --eval-freq 20000 --eval-horizon 250 --num-eval-episodes 20
$S/policy/ddim_copy.py checkpoints/stackcube/policy_120k/best.pt     # -> best_ddim10.pt (selected update 100k)
$S/policy/ddim_copy.py checkpoints/liftpeg/policy_120k/best.pt       # -> best_ddim10.pt (selected update 120k)
```

## 4. PickCube corrective fine-tuning (report: Section 4.5, Table 4)

Each stage rolls out the current policy under pushes, lets a scripted expert take over from states where the policy
failed, and fine-tunes for 5,000 updates on the nominal demonstrations plus all corrections collected so far
(learning rate 1e-5, batch 32). The teachers are in `src/adaptive_action_chunking/corrections/`.

| Stage | New corrections (`data/`) | Initialised from | Output (`checkpoints/pickcube/`) |
| --- | --- | --- | --- |
| 0 nominal | - | scratch | `stage0_nominal` |
| 1 placement, then pickup | `recovery`, `recovery_round2`, `recovery_pickup` | 0 | `stage1_placement`, then `stage1_pickup` |
| 2 more pickup | `recovery_pickup_round2` | 1 | `stage2_pickup` |
| 3 grasp closing | `recovery_alignment`, `recovery_learner`, then `recovery_closing` | 2 | `stage3_branch`, then `stage3_closing` |
| 4 direct descent | `recovery_smooth`, `recovery_descent` | 3 | `stage4_descent` |
| 5 level alignment | `recovery_level` | 4 | `stage5_level` (monitor-study policy) |
| second policy | all ten sets above | `seed2_nominal` | `seed2_finetuned` |

Stages 1 and 3 each take two fine-tunes. In stage 3 the first fine-tune (`stage3_branch`) adds approach and retry
corrections from learner states; its rollouts supply the ready-to-close states for the closing corrections, and the
stage-3 policy is then fine-tuned from stage 2 on all of them.
The early collectors were edited between rounds, so these commands repeat the procedure but do not regenerate
byte-identical datasets.

```bash
R="data/recovery data/recovery_round2 data/recovery_pickup"
# stage 1: placement corrections, a second placement round, then pickup corrections
$S/corrections/collect_recovery_demos.py --policy checkpoints/pickcube/stage0_nominal/best.pt --workers 8
$S/policy/finetune_policy.py --sampler ddpm --output-dir checkpoints/pickcube/stage1_placement
$S/corrections/collect_recovery_demos.py --policy checkpoints/pickcube/stage1_placement/best.pt \
  --output-dir data/recovery_round2 --seed 90273 --workers 8 --forces 0 0.3 0.4 0.5
$S/corrections/collect_recovery_demos.py --teacher pickup --policy checkpoints/pickcube/stage1_placement/best.pt \
  --output-dir data/recovery_pickup --seed 90302 --workers 8 --forces 0.4 0.5
$S/policy/finetune_policy.py --sampler ddpm --init-checkpoint checkpoints/pickcube/stage1_placement/best.pt \
  --recovery-dir $R --recovery-fraction 0.25 --output-dir checkpoints/pickcube/stage1_pickup
# stage 2: 128/32 more pickup corrections
$S/corrections/collect_recovery_demos.py --teacher pickup --policy checkpoints/pickcube/stage1_pickup/best.pt \
  --output-dir data/recovery_pickup_round2 --train-count 128 --validation-count 32 --seed 90305 --workers 8 --forces 0.4 0.5
$S/policy/finetune_policy.py --sampler ddpm --init-checkpoint checkpoints/pickcube/stage1_pickup/best.pt \
  --recovery-dir $R data/recovery_pickup_round2 --balance-recovery-sources \
  --output-dir checkpoints/pickcube/stage2_pickup
R="$R data/recovery_pickup_round2 data/recovery_alignment data/recovery_learner"
# stage 3, first fine-tune: learner-state approach/retry corrections (seeds 90307, 90309); supplies the closing states
$S/corrections/collect_grasp_alignment.py
$S/corrections/collect_learner_corrections.py --pilot
$S/corrections/collect_learner_corrections.py
$S/policy/finetune_policy.py --sampler ddpm --init-checkpoint checkpoints/pickcube/stage2_pickup/last.pt \
  --recovery-dir $R --recovery-fraction 0.125 --output-dir checkpoints/pickcube/stage3_branch
# stage 3, second fine-tune: ready-to-close corrections (seed 90310)
$S/corrections/collect_closing_corrections.py
$S/policy/finetune_policy.py --sampler ddpm --init-checkpoint checkpoints/pickcube/stage2_pickup/last.pt \
  --recovery-dir $R data/recovery_closing --recovery-fraction 0.125 --output-dir checkpoints/pickcube/stage3_closing
R="$R data/recovery_closing data/recovery_smooth data/recovery_descent"
# stage 4: smoother teacher (seed 90312) and direct descent from aligned states (seed 90313); DDIM10 selection from here on
$S/corrections/collect_smooth_corrections.py
$S/corrections/collect_smooth_corrections.py --descent
$S/policy/finetune_policy.py --init-checkpoint checkpoints/pickcube/stage3_closing/last.pt \
  --recovery-dir $R --recovery-fraction 0.125 --output-dir checkpoints/pickcube/stage4_descent
# stage 5: level alignment at the current height, then descend
$S/corrections/collect_smooth_corrections.py --level
$S/policy/finetune_policy.py --init-checkpoint checkpoints/pickcube/stage4_descent/best.pt \
  --recovery-dir $R data/recovery_level --recovery-fraction 0.125 --focus-latest-queries \
  --output-dir checkpoints/pickcube/stage5_level
# second policy: one fine-tune of the seed-2 nominal policy on all ten correction sets
$S/policy/finetune_policy.py --init-checkpoint checkpoints/pickcube/seed2_nominal/best.pt \
  --recovery-dir $R data/recovery_level --recovery-fraction 0.125 --focus-latest-queries --seed 1 \
  --output-dir checkpoints/pickcube/seed2_finetuned
```

**Policy evaluations.**

- *Closing confirmation:* 100 groups x 0/0.2/0.3/0.4 N, K8/K15, 150 steps, DDPM100, seed 169001. Stage 2 is
  compared with stage 3; the result is +4.25 points [+1.75, +6.75] at K8. Commands:

  ```bash
  $S/policy/closing_confirmation/prepare.py
  $S/policy/evaluate_policy.py --compare-policies retained=checkpoints/pickcube/stage2_pickup/last.pt \
    closing=checkpoints/pickcube/stage3_closing/last.pt --baseline retained --groups 100 --seed 169001 --workers 8 \
    --output results/policy_closing_confirmation/comparison
  $S/policy/closing_confirmation/audit_rollouts.py
  $S/policy/extend_policy_rollouts.py --source results/policy_closing_confirmation/comparison/results.json \
    --output-dir results/policy_closing_confirmation/horizon150
  $S/policy/closing_confirmation/audit_rollouts.py --extended
  $S/policy/closing_confirmation/summarize.py
  ```

- *Lineage benchmark:* 30 groups x 0-0.5 N, K8/K15, 150 steps, DDIM10, seed 199001. Its cases are re-run for
  every stage in Table 4. Commands:

  ```bash
  $S/policy/compare_policy_updates.py     # stage 4 vs stage 5 (+ stage 3) -> results/policy_level_eval
  $S/analysis/sampler_table.py            # Table 4, DDIM10 columns -> results/sampler_table/
  $S/analysis/sampler_table.py --ddpm stage0 stage1 stage4 stage5 seed2_nominal seed2_finetuned   # DDPM100 columns
  ```

## 5. LiftPegUpright corrections with a planner teacher (report: Section 4.4)

The base policy had no fixed-K headroom: shorter chunks hardly helped after a push. The commands below are the
pilots behind that finding. ManiSkill's planner then takes over 2-10 steps after a 2-4 N push. Its joint actions are
converted to `pd_ee_delta_pose` and kept as correction segments (planner success 360/400; 287 train /
62 validation kept).

```bash
export ROBOTS_TASK=LiftPegUpright P=checkpoints/liftpeg/policy_120k/best_ddim10.pt
$S/monitor/pilot_fixed_k.py --policy $P --output-dir results/liftpeg_nominal_check --groups 40 --seed 389101 \
  --chunks 8 10 15 --forces 0 --horizon 250
$S/monitor/pilot_fixed_k.py --policy $P --output-dir results/liftpeg_force_pilot --groups 30 --seed 389201 \
  --chunks 8 15 --forces 0 1 2 3 4 6 --horizon 250
$S/corrections/planner_corrections.py states --policy $P --output-dir data/liftpeg_recovery_states \
  --count 400 --seed 389501 --forces 2 3 4
.venv-mp/bin/python scripts/corrections/planner_corrections.py teach --states-dir data/liftpeg_recovery_states \
  --output-dir data/liftpeg_recovery
$S/policy/finetune_policy.py --init-checkpoint checkpoints/liftpeg/policy_120k/best.pt --nominal-dir data/liftpeg/split \
  --recovery-dir data/liftpeg_recovery --total-iters 10000 --selection-horizon 250 --output-dir checkpoints/liftpeg/policy_01_recovery
$S/monitor/pilot_fixed_k.py --policy $P checkpoints/liftpeg/policy_01_recovery/last.pt \
  --output-dir results/liftpeg_recovery_pilot --groups 40 --seed 389601 --chunks 8 15 --forces 0 3 4 --horizon 250 --workers 10
```

A second round collects takeovers from the round-1 policy (planner success 355/400; 286 train / 59 validation kept).
It added nothing. Its pilot re-runs the base and round-1 policies on fresh groups, and the report's 3 N K8-vs-K15
numbers (Section 4.4) come from it.

```bash
P1=checkpoints/liftpeg/policy_01_recovery/last.pt
$S/corrections/planner_corrections.py states --policy $P1 --output-dir data/liftpeg_recovery_round2_states \
  --count 400 --seed 389701 --forces 3 3.5 4
.venv-mp/bin/python scripts/corrections/planner_corrections.py teach --states-dir data/liftpeg_recovery_round2_states \
  --output-dir data/liftpeg_recovery_round2
$S/policy/finetune_policy.py --init-checkpoint $P1 --nominal-dir data/liftpeg/split \
  --recovery-dir data/liftpeg_recovery data/liftpeg_recovery_round2 --recovery-fraction 0.5 --total-iters 10000 \
  --selection-horizon 250 --output-dir checkpoints/liftpeg/policy_02_recovery_round2
$S/monitor/pilot_fixed_k.py --policy $P $P1 checkpoints/liftpeg/policy_02_recovery_round2/{best,last}.pt \
  --output-dir results/liftpeg_recovery_round2_pilot --groups 40 --seed 389801 --chunks 8 15 --forces 0 3 4 --horizon 250 --workers 10
```

The monitor-study policy is `checkpoints/liftpeg/policy_01_recovery/last.pt` (update 10k, DDIM10).

## 6. Monitor study (report: Sections 3 and 4.2, Table 2, Figure 2)

Pushes are two-step horizontal pulses on the object, starting at a random step in 8-18 (approach phase).
Groups are split once before any outcome is seen: 40 train (predictor data), 20 predictor validation,
30 scheduler validation (threshold selection) and 40 test.

- **Triggers.** Acceleration: second difference of the object position. Prediction error: residual of an MLP
  (`16->128->128->6`, 19 inputs with 7-D actions) that predicts the next object and gripper positions.
  The predictor is trained on nominal K8/K15 rollouts.
- **Threshold selection.** On scheduler validation, each trigger takes the threshold whose mean calls fall within
  +-10% of the fixed K15/K10/K8 budgets, with the lowest time to goal (ties: higher success).
- **Confirmation.** A random-interruption control is calibrated on validation groups to the trigger's call count.
  A second fresh cohort is then run, and both cohorts are pooled. All intervals resample whole reset groups
  (5,000 draws).

`study.py run` collects predictor data and trains the predictor, `validate` selects thresholds, and `test` runs the
frozen selection once.

```bash
mstudy() {  # task-independent pipeline: policy, data dir, predictor dir, results dir, collection seed, horizon
  $S/monitor/study.py run --policy $1 --dataset-dir $2 --output-dir $3 --collection-seed $5 --horizon $6 --workers 8
  for stage in validate test; do $S/monitor/study.py $stage --dataset-dir $2 --output-dir $3 --results-dir $4 --workers 8; done
}
mconfirm() {  # data dir, predictor dir, results dir, confirmation dir, confirmation groups, seed, extra flags
  A="--dataset-dir $1 --predictor $2/predictor.pt --selection $3/selection.json --test-results $3/test.json --output-dir $4 --workers 8 $7"
  $S/monitor/confirm.py calibrate $A; $S/monitor/confirm.py control $A
  $S/monitor/confirm.py confirm $A --groups $5 --seed $6; $S/monitor/confirm.py pool $A
}
# PickCube (forces 0/0.3/0.4/0.5 N, 150 steps)
mstudy checkpoints/pickcube/stage5_level/best.pt data/monitor_level checkpoints/pickcube/monitor_level \
  results/monitor_level 33101 150
mconfirm data/monitor_level checkpoints/pickcube/monitor_level results/monitor_level results/monitor_confirmation 40 339001
# PickCube second policy
mstudy checkpoints/pickcube/seed2_finetuned/best.pt data/monitor_seed2 checkpoints/pickcube/monitor_seed2 \
  results/monitor_seed2 33102 150
mconfirm data/monitor_seed2 checkpoints/pickcube/monitor_seed2 results/monitor_seed2 results/monitor_seed2_confirmation 40 411004
# StackCube (0/0.2/0.3/0.4 N, 200 steps)
export ROBOTS_TASK=StackCube
mstudy checkpoints/stackcube/policy_120k/best_ddim10.pt data/monitor_stack checkpoints/stackcube/monitor results/monitor_stack 53101 200
mconfirm data/monitor_stack checkpoints/stackcube/monitor results/monitor_stack results/monitor_stack_confirmation 60 379001 \
  "--targets acceleration_calls14 acceleration_calls25"
# LiftPegUpright (0/2/3/3.5 N, 250 steps)
export ROBOTS_TASK=LiftPegUpright
mstudy checkpoints/liftpeg/policy_01_recovery/last.pt data/monitor_liftpeg checkpoints/liftpeg/monitor results/monitor_liftpeg 53201 250
mconfirm data/monitor_liftpeg checkpoints/liftpeg/monitor results/monitor_liftpeg results/monitor_liftpeg_confirmation 60 389901 \
  "--targets acceleration_calls25"
unset ROBOTS_TASK
```

Outputs: `results/monitor_*_confirmation/pooled.json` (Table 2: pooled test + confirmation groups).

## 7. Robustness cohorts and fixed-K frontier (report: Sections 4.2-4.3, Table 3, Figures 3-4)

Each cohort uses fresh groups with the frozen selection and the calibrated random control copied from Section 6.

```bash
for d in grasp_phase grasp_phase_b long_pulse; do mkdir -p results/monitor_$d; cp results/monitor_confirmation/calibration.json results/monitor_$d/; done
$S/monitor/confirm.py confirm --output-dir results/monitor_grasp_phase   --groups 40 --seed 411001 --start-range 30 60
$S/monitor/confirm.py confirm --output-dir results/monitor_grasp_phase_b --groups 40 --seed 411002 --start-range 30 60
$S/monitor/confirm.py confirm --output-dir results/monitor_long_pulse    --groups 40 --seed 411003 --duration 5
$S/analysis/summarize_variant_cohorts.py
$S/monitor/confirm.py sweep        # PickCube K1/K2/K4/K6 on the pooled groups; K15 rerun must match exactly

export ROBOTS_TASK=StackCube
for d in stack_carry stack_long_pulse; do mkdir -p results/monitor_$d; cp results/monitor_stack_confirmation/calibration.json results/monitor_$d/; done
A="--dataset-dir data/monitor_stack --predictor checkpoints/stackcube/monitor/predictor.pt --selection results/monitor_stack/selection.json \
   --test-results results/monitor_stack/test.json --targets acceleration_calls14 acceleration_calls25 --workers 8"
$S/monitor/confirm.py confirm $A --output-dir results/monitor_stack_carry --groups 40 --seed 421005 --start-range 60 90
$S/monitor/confirm.py confirm $A --output-dir results/monitor_stack_long_pulse --groups 40 --seed 421006 --duration 5
$S/monitor/confirm.py sweep $A --output-dir results/monitor_stack_confirmation
unset ROBOTS_TASK
```

## 8. Analyses, figures and the reproduction check

None of these steps runs new training. Only the reproduction check and `bob_stats.py`'s input regenerate rollouts.

```bash
$S/analysis/latency_sensitivity.py      # blocking-inference sensitivity -> results/latency_sensitivity.json
$S/analysis/trigger_comparison.py       # trigger vs predictor, random-control timing -> results/trigger_comparison.json
$S/analysis/make_figures.py             # report figures as PNGs in results/figures/ (--report: PDFs in results/figures/report/)
$S/analysis/bob_stats.py results/monitor_level/test.json results/bob_stats.json   # needs full traces (rerun the test stage)
$S/analysis/reproduce_results.py --workers 12          # re-run every saved rollout; reports exact matches
$S/analysis/reproduce_results.py stage5 --groups 2     # quick smoke test (one policy, first two groups per cohort)
$S/analysis/export_report_results.py    # Section 9 files + per-episode table -> report_results/ (tracked)
```

The full reproduction check re-ran all 22,780 cube-task rollouts behind the report and matched each one exactly. The
LiftPegUpright cohorts were added to the check later (`liftpeg` key).

## 9. Where each report result comes from

Every file in the table below, plus each main cohort's `selection.json` (thresholds) and `calibration.json` (random
control), is copied to `report_results/summaries/` under the same relative path, with per-episode lists removed.
`report_results/episodes.csv` has one row per monitor-study episode in every cohort of Tables 2-3 and Figures 2-4,
checked against the saved summaries on export: cohort, split (test or fresh groups), reset group and seeds, pulse
force/start/duration, variant, policy calls, trigger replans, time to goal (s, horizon if never reached), goal step,
final success and the steps at which the policy was called. Rows sharing `group_id` within a cohort are paired. The
two PickCube grasp-phase cohorts are pooled in Table 3, and `*_exploratory` variants are not reported.

| Report item | Result file(s) under `results/` |
| --- | --- |
| Table 2 (main results) | `monitor_confirmation/pooled.json`, `monitor_seed2_confirmation/pooled.json`, `monitor_stack_confirmation/pooled.json`, `monitor_liftpeg_confirmation/pooled.json` |
| Table 3, Figure 4 (robustness) | `monitor_{long_pulse,grasp_phase,grasp_phase_b,stack_carry,stack_long_pulse}/confirmation.json` |
| Figure 3 (fixed-K frontier) | `monitor_confirmation/fixed_k_sweep.json`, `monitor_stack_confirmation/fixed_k_sweep.json` |
| Acceleration vs prediction-error trigger (Section 4.2), random-control timing check (Section 3), test vs fresh split (Section 4.2) | `trigger_comparison.json` |
| Inference-time paragraph | `latency_sensitivity.json` |
| Table 4 (policy lineage) | `sampler_table/sampler_table.json`, `policy_closing_confirmation/summary.json` |
| LiftPegUpright pilots | `liftpeg_force_pilot/`, `liftpeg_recovery_pilot/`, `liftpeg_recovery_round2_pilot/` (3 N K8 vs K15 in Section 4.4: `liftpeg_recovery_round2_pilot/`) |
