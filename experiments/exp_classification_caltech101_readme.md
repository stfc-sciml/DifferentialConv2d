1. Train models:

    ```bash
    python exp_classification_caltech101.py --conv2d-method Zero --seed 0 --device cuda
    ```
    
    where `--conv2d-method` can be
    
    * `'Zero'`: padding by zeros;
    * `'Repl'`: padding by replication;
    * `'Refl'`: padding by reflection;
    * `'Circ'`: padding by circular;
    * `'Rand'`: padding by random distribution;
    * `'Extr'`: padding by extrapolation;
    * `'Part'`: partial convolution;
    * `'EBH'`: original Explicit Boundary Handling;
    * `'Diff'`: our differential method;
    * `'Diff-EBH'`: our differential method with EBH.
    
    Use `--seed` to vary model initialization. 
    Our paper reports the mean accuracy for seeds `[0, 1, 2, 3, 4]`.
    
    Data will be downloaded to `./exp_data/caltech101/` upon the first run.
    
    Training history and model weights will be saved to `./exp_results/caltech101/`.

2. Show the accuracy and training history:
    ```bash
    python exp_classification_caltech101_metrics.py --seed 0
    ```