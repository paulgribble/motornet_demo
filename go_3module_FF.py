import json
from reaching_model import ReachingModel
from my_utils import model_path

FF_STRENGTH = 10.0   # strength of the clockwise curl force field (N.s/m)

# Example: 3-module network (PMd,M1,SC), trained in the null field (NF) and then in a
# clockwise curl force field (CWFF). After the main NF training some of the weights are fixed. See visualize_modular_FF.ipynb for the analysis.
model = ReachingModel.create(
    "my_3mod_FF",
    module_names = ["premotor", "motor", "spinal"],
    module_sizes = [128, 128, 32],
    vision_mask  = [1.0, 1.0, 0.0],
    proprio_mask = [0.0, 0.0, 1.0],
    task_mask    = [1.0, 0.0, 0.0],
    # row = receiving module, column = sending module; diagonal = recurrence within a module
    connectivity_mask = [
        # PMd   M1    SC
        [ 0.7,  0.2,  0.0 ],   # PMd receives from M1
        [ 0.5,  0.7,  0.1 ],   # M1 receives from PMd, SC
        [ 0.0,  0.5,  0.7 ],   # SC receives from M1
    ],
    output_mask  = [0.0, 0.0, 1.0],
)

# 1. NULL FIELD (NF) TRAINING
# fast=True uses the equivalent but ~3x faster simulation in fast_rollout.py (it takes a minute or two to
# compile the first time, so it is only worth it for long runs like this one)
model.train(n_batches=20000, batch_size=32, fast=True, plot_interval=1000)

# From here on these weights stay fixed: spinal -> motor, spinal -> spinal, inputs (vision, proprioception, task) -> hidden,
# hidden -> muscles, and all the biases. Everything else (the other weights between hidden units) can
# still learn.
model.freeze(connections=[("spinal", "motor"), ("spinal", "spinal")], inputs=True, outputs=True, hidden_biases=True)

model.train(n_batches=500, batch_size=8, task_mode='center_out')
model.test(n_targets=8)

# Keep a copy of the network as it is at the end of NF training, to compare with the end of FF learning
model.save_checkpoint("NF")
with open(model_path(model.name, "phases.json"), 'w') as f:
    json.dump({"ff_strength": FF_STRENGTH}, f)   # the analysis notebook tests the network in the same field

# 2. FORCE FIELD (CWFF) LEARNING
# (the fast simulation is compiled again here, because the force field is now on)
model.train(n_batches=5000, batch_size=32, ff_strength=FF_STRENGTH, fast=True, plot_interval=1000)
model.train(n_batches=500, batch_size=8, ff_strength=FF_STRENGTH, task_mode='center_out')
model.test(n_targets=8, ff_strength=FF_STRENGTH)
model.save()
