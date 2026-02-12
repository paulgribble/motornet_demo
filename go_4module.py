from reaching_model import ReachingModel

# Example: 4-module network (PMd,M1,S1,SC)
model = ReachingModel.create(
    "my_4mod",
    n_modules    = 4,
    module_names = ["premotor", "motor", "sensory", "spinal"],
    module_sizes = [128, 128, 64, 32],
    vision_mask  = [1.0, 1.0, 0.0, 0.0],
    proprio_mask = [0.0, 0.0, 0.1, 1.0],
    task_mask    = [1.0, 0.0, 0.0, 0.0],
    connectivity_mask=[
                    # PMd   M1    S1    SC
                    [ 1.00,  0.20,  0.00,  0.00 ],  # PMd receives from M1
                    [ 0.20,  1.00,  0.20,  0.10 ],  # M1 receives from PMd, S1, SC
                    [ 0.00,  0.20,  1.00,  1.00 ],  # S1 receives from M1, SC
                    [ 0.00,  0.20,  0.00,  1.00 ],  # SC receives from M1
    ],
    output_mask=     [0.0, 0.0, 0.0, 1.0],
)

model.train(n_batches=10000, batch_size=32)
model.train(n_batches=500, batch_size=8, task_mode='center_out')
model.test(n_targets=8)
model.save()
