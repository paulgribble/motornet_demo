from reaching_model import ReachingModel

# Example: 3-module network (PMd,M1,SC)
model = ReachingModel.create(
    "my_3mod",
    n_modules    = 3,
    module_names = ["premotor", "motor", "spinal"],
    module_sizes = [128, 128, 32],
    vision_mask  = [1.0, 1.0, 0.0],
    proprio_mask = [0.0, 0.0, 1.0],
    task_mask    = [1.0, 0.0, 0.0],
    connectivity_mask=[
                    #  PMd   M1    SC
                    [ 0.7,  0.2,  0.0 ],   # PMd receives from M1
                    [ 0.5,  0.7,  0.1 ],   # M1 receives from PMd
                    [ 0.0,  0.5,  0.7 ],   # SC receives from M1
    ],
    output_mask=     [0.0, 0.0, 1.0],
)

model.train(n_batches=10000, batch_size=32)
model.train(n_batches=500, batch_size=8, task_mode='center_out')
model.test(n_targets=8)
model.save()
