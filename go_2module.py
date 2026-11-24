from reaching_model import ReachingModel

# Example: 2-module network (M1,SC)
# (these are also ReachingModel's defaults, written out here for comparison with the other examples)
model = ReachingModel.create(
    "my_2mod",
    module_names = ["motor", "spinal"],
    module_sizes = [128, 32],
    vision_mask  = [1.0, 0.0],
    proprio_mask = [0.0, 1.0],
    task_mask    = [1.0, 0.0],
    # row = receiving module, column = sending module; diagonal = recurrence within a module
    connectivity_mask = [
        # M1    SC
        [ 0.7,  0.1 ],   # M1 receives from SC
        [ 0.5,  0.7 ],   # SC receives from M1
    ],
    output_mask  = [0.0, 1.0],
)

# fast=True uses the equivalent but ~3x faster simulation in fast_rollout.py (it takes a minute or two to
# compile the first time, so it is only worth it for long runs like this one)
model.train(n_batches=10000, batch_size=32, fast=True)
model.train(n_batches=500, batch_size=8, task_mode='center_out')
model.test(n_targets=8)
model.save()
