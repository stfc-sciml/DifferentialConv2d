1. Download the dataset following [torchvision.datasets.Cityscapes](https://pytorch.org/vision/main/generated/torchvision.datasets.Cityscapes.html).
Put `gtFine/` and `leftImg8bit/` under `./exp_data/cityscapes/`.

2. Train models:

    ```bash
    python exp_segmentation_cityscapes.py --conv2d-method Zero --seed 0 --device cuda --downscale 2
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
    
    Use `--downscale` to reduce image size. The original image size is 1024x2048. 
    `--downscale 2` will reduce the size to 512x1024.
    
    Training history and model weights will be saved to `./exp_results/cityscapes/`.

3. Show the segmentation metrics and training history:
    ```bash
    python exp_segmentation_cityscapes_metrics.py --seed 0
    ```