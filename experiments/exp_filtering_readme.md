1. Download `NS_Re500_s256_T100_test.npy` from https://github.com/neuraloperator/physics_informed and put it under `exp_data/NS/`. 
It contains 100 solutions to the Navier-Stokes equations with different initial conditions.

2. Compute the L1 errors for the three datasets (Chebyshev, spherical harmonics and Navier-Stokes):

    ```bash
    python exp_filtering_compute.py
    ```
    
3. Visualize the results:
    ```bash
    python exp_filtering_plot.py
    ```
   This will generate `exp_results/filtering/filter.pdf`, which is Figure 2 in our paper.