from reaching_model import ReachingModel

# Example: 2-module network (M1,SC)
model = ReachingModel.create(
    "my_2mod",
    n_modules    = 2,
    module_names = ["motor", "spinal"],
    module_sizes = [128, 32],
    vision_mask =  [1.0, 0.0],
    proprio_mask = [0.0, 1.0],
    task_mask =    [1.0, 0.0],
    connectivity_mask =[
                    # M1    SC
                    [ 0.7,  0.1 ],   # M1 receives from SC
                    [ 0.5,  0.7 ],   # SC receives from M1
    ],
    output_mask =   [0.0, 1.0],
)

model.train(n_batches=10000, batch_size=32)
model.train(n_batches=500, batch_size=32, task_mode='center_out')
model.test(n_targets=8)
model.save()
