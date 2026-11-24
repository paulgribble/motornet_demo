from reaching_model import ReachingModel

# Example: 4-module network (PMd,M1,S1,SC)
model = ReachingModel.create(
    "my_4mod",
    module_names = ["premotor", "motor", "sensory", "spinal"],
    module_sizes = [128, 128, 64, 32],
    vision_mask  = [1.0, 1.0, 0.0, 0.0],
    proprio_mask = [0.0, 0.0, 0.1, 1.0],
    task_mask    = [1.0, 0.0, 0.0, 0.0],
    # row = receiving module, column = sending module; diagonal = recurrence within a module
    connectivity_mask = [
        # PMd    M1     S1     SC
        [ 1.00,  0.20,  0.00,  0.00 ],   # PMd receives from M1
        [ 0.20,  1.00,  0.20,  0.10 ],   # M1 receives from PMd, S1, SC
        [ 0.00,  0.20,  1.00,  1.00 ],   # S1 receives from M1, SC
        [ 0.00,  0.20,  0.00,  1.00 ],   # SC receives from M1
    ],
    output_mask  = [0.0, 0.0, 0.0, 1.0],
)

# fast=True uses the equivalent but ~3x faster simulation in fast_rollout.py (it takes a minute or two to
# compile the first time, so it is only worth it for long runs like this one)
model.train(n_batches=10000, batch_size=32, fast=True, plot_interval=1000)
model.train(n_batches=500, batch_size=8, task_mode='center_out')
model.test(n_targets=8)
model.save()
