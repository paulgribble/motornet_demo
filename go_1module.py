from reaching_model import ReachingModel

# Example: 1-module network (M1)
model = ReachingModel.create(
    "my_1mod",
    n_modules    = 1,
    module_names = ["motor"],
    module_sizes = [128],
    vision_mask =  [1.0],
    proprio_mask = [1.0],
    task_mask =    [1.0],
    connectivity_mask =[
                    # M1
                    [ 0.7 ],   # M1 receives from M1 (recurrence)
    ],
    output_mask =   [1.0],
)

model.train(n_batches=10000, batch_size=32)
model.train(n_batches=500, batch_size=8, task_mode='center_out')
model.test(n_targets=8)
model.save()
