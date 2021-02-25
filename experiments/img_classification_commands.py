model_ds_combs = [('MLP', 'MNIST'),
                  ('CNN', 'MNIST'),
                  ('MLP', 'FMNIST'),
                  ('CNN', 'FMNIST'),
                  ('CNN', 'CIFAR10'),
                  ('AllCNN', 'CIFAR10')]

seeds = [117, 68, 187, 27, 51]

for model, ds in model_ds_combs:
    print(f'###### {model} on {ds} ######')
    for seed in seeds:
        # train models
        cmd = f'python imgclassification.py -d {ds} -m {model} -s {seed}'
        print(cmd)
        # inference and performance estimation
        base_cmd = f'python imginference.py -d {ds} -m {model} --seed {seed} --loginfo'
        # predictive comparison
        print(base_cmd)
        # OOD performance
        print(base_cmd + ' --ood')
        # GP performance
        print(base_cmd + ' --gp')
