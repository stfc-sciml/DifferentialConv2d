1. Train models:

    ```bash
    python exp_super_resolution_etopo.py --conv2d-method Zero --seed 0 --device cuda
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
    Our paper reports the mean metrics for seeds `[0, 1, 2, 3, 4]`.
    
    Data will be downloaded to `./exp_data/etopo/` upon the first run, taking ~5GB storage.
    
    Training history and model weights will be saved to `./exp_results/etopo/`.

2. Print the errors for different `conv2d` methods and validation patch sizes:
    ```bash
    python exp_super_resolution_etopo_metrics.py --seed 0
    ```
   
3. Visualize the error maps in an area:
    ```bash
    python exp_super_resolution_etopo_plot.py
    ```
   This will generate `exp_results/etopo/etopo.pdf`, which is Figure 3 in our paper.