datasets = ['australian', 'breast_cancer', 'digits', 'glass',
            'ionosphere', 'satellite', 'vehicle', 'waveform']
seeds = [711, 1, 75, 359, 17, 420, 129, 666, 69, 36]

for ds in datasets:
    for seed in seeds:
        cmd_single = f'python classification.py -d {ds} --seed {seed} --n_layers 1 --activation tanh --name tanh_1'
        print(cmd_single)
        cmd_double = f'python classification.py -d {ds} --seed {seed} --n_layers 2 --activation tanh --name tanh_2'
        print(cmd_double)
