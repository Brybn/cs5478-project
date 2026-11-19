# When to replan: event-triggered action chunking for diffusion policies

A diffusion policy predicts a chunk of up to 15 actions and executes it open-loop. If the object is pushed during
the chunk, the robot keeps following a stale plan. This project cuts a chunk short when a cheap signal, the
acceleration of the manipulated object, shows that the scene has changed. We compare this trigger with fixed chunk
lengths K and with a random-interruption control that makes the same number of policy calls. The experiments use
three ManiSkill 3 tasks: PickCube, StackCube and LiftPegUpright.

## Main result

Time to goal under approach-phase pushes (simulated seconds; acceleration trigger minus baseline; 95% interval from
a reset-group bootstrap; all with 10-step DDIM sampling):

| Cohort | vs fixed K15 | vs random control |
| --- | --- | --- |
| PickCube (80 groups x 4 forces) | -0.32 [-0.43, -0.21] | -0.21 [-0.33, -0.10] |
| PickCube, second policy (80) | -0.43 [-0.57, -0.29] | -0.36 [-0.49, -0.23] |
| StackCube (100) | -0.54 [-0.66, -0.43] | -0.45 [-0.58, -0.34] |
| LiftPegUpright (100) | -0.27 [-0.41, -0.13] | -0.24 [-0.37, -0.10] |

The gain disappears once the object is held (grasp-phase and carry-phase pushes). Replanning at every step (K1) is
the slowest schedule. If the robot must wait for every policy call, the PickCube gain is no longer significant.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/adaptive_action_chunking/policy.py` | Conditional 1-D UNet diffusion policy (adapted from the ManiSkill baseline), DDPM/DDIM sampling, checkpoint loading |
| `src/adaptive_action_chunking/env.py` | Task selection (`ROBOTS_TASK`), push-disturbed environments, policy and monitor observations |
| `src/adaptive_action_chunking/train.py`, `finetune.py` | Behaviour cloning; corrective fine-tuning on demonstrations plus expert corrections |
| `src/adaptive_action_chunking/evaluate.py` | Rollouts with fixed or trigger-cut chunks, paired group-bootstrap summaries, threshold selection |
| `src/adaptive_action_chunking/monitor.py` | Acceleration and prediction-error triggers, predictor training, seed planning |
| `src/adaptive_action_chunking/corrections/` | Scripted PickCube correction teachers and the correction-collection pipeline |
| `scripts/demos/` | Demonstration download/generation, replay and train/validation split |
| `scripts/policy/` | Policy training, fine-tuning, DDIM inference copies, policy-lineage evaluations |
| `scripts/corrections/` | Collection of correction datasets (scripted teachers for PickCube, motion planner for LiftPegUpright) |
| `scripts/monitor/` | Monitor study: predictor data and training, threshold selection, test, random control, confirmation |
| `scripts/analysis/` | Report figures, inference-time sensitivity, sampler table, exact reproduction check, result export |
| `report_results/` | Summary results behind every number in the report, and one row per monitor-study episode (`episodes.csv`) |

## Running the code

Requirements: Ubuntu or WSL2, Python 3.10, [uv](https://docs.astral.sh/uv/). Training used one CUDA GPU; rollouts and
evaluation run on CPU.

```bash
uv sync --frozen --python 3.10
sudo apt-get install libvulkan1 mesa-vulkan-drivers      # CPU rendering, if not already installed
```

`report_results/` holds the summary results behind every number in the report; workflow section 9 maps each
table and figure to its file. The trained checkpoints, datasets and full rollouts (several GB) are not in the
repository and are available on request. Section numbers below refer to [docs/workflow.md](docs/workflow.md), which
lists every command, seed and output path. A minimal PickCube run
from scratch:

```bash
S="uv run --frozen --no-sync python scripts"
# 1. demonstrations (workflow section 2)
uv run --frozen --no-sync python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos
$S/demos/replay_demos.py --traj-path data/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state --save-traj -b physx_cpu --count 100
$S/demos/prepare_demos.py --source data/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
# 2. nominal policy and its DDIM10 inference copy (section 3)
$S/policy/train_policy.py --output-dir checkpoints/pickcube/demo_policy
$S/policy/ddim_copy.py checkpoints/pickcube/demo_policy/best.pt
# 3. how much do shorter fixed chunks help under pushes? (headroom pilot)
$S/monitor/pilot_fixed_k.py --policy checkpoints/pickcube/demo_policy/best_ddim10.pt --output-dir results/demo_pilot
# 4. monitor study: predictor, threshold selection on validation groups, frozen test (section 6)
$S/monitor/study.py run --policy checkpoints/pickcube/demo_policy/best_ddim10.pt --dataset-dir data/demo_monitor \
  --output-dir checkpoints/pickcube/demo_monitor --collection-seed 33101 --horizon 150 --workers 8
$S/monitor/study.py validate --dataset-dir data/demo_monitor --output-dir checkpoints/pickcube/demo_monitor \
  --results-dir results/demo_monitor --workers 8
$S/monitor/study.py test --dataset-dir data/demo_monitor --output-dir checkpoints/pickcube/demo_monitor \
  --results-dir results/demo_monitor --workers 8
```

A nominal policy trained from demonstrations alone is fragile after a push. In the report, the PickCube monitor
study uses a policy after five rounds of corrective fine-tuning (workflow section 4). Every saved rollout records its
seeds, so a re-run with the same checkpoint reproduces it exactly. `scripts/analysis/reproduce_results.py` checks
this.

## Acknowledgements

The diffusion-policy architecture and the demonstration replay/generation scripts adapt the
[ManiSkill v3.0.1 diffusion-policy baseline](https://github.com/mani-skill/ManiSkill/tree/v3.0.1/examples/baselines/diffusion_policy)
and tools (Apache-2.0; license in `third_party/MANISKILL_LICENSE`).
