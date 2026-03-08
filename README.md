# MDA Influence 

EK-FAC based influence score computation for attention heads in TransformerLens models.
Supports QK-only and QKVO subspaces with configurable probes.
Currently, the probe supports copy_target (induction head) and prev_attn (previous token head). 
Probes for other components can be customized.

## Quick Start

```bash
# Single GPU
python scripts/run_influence.py --config config/prev_attn.yaml

# Multi-GPU (torchrun)

torchrun --nproc_per_node=4 scripts/run_influence.py --config config/copy_target.yaml

##Configuration:
target:
  layer: 5
  head: 10
  mode: "qkvo"   # "qk" or "qkvo"

probe:
  type: "copy_target_synthetic"   # "copy_target_synthetic" | "copy_target_dataset" | "prev_attn"
  num_samples: 32
  induction_match: "current"
  match_choice: "last"

output:
  dir: "/root/results"
  top_k: 1536

##Project Structure:
  core/        # EK-FAC blocks, stage1A/1B, influence scoring
  model/       # Hook caches and hook setup
  probes/      # Probe functions (copy_target, prev_attn)
  data/        # NPY dataloaders
  config/      # YAML configs
  scripts/     # CLI entry
  utils/       # Utilities


##Output:
Two numpy arrays saved to output.dir:

top_pos.npy — most positively influential samples, shape [top_k, 3] (score, index, loss)

top_neg.npy — most negatively influential samples, shape [top_k, 3]