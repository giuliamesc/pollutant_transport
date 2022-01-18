# Pollutant Transport Rare Events :national_park:
This repo contains the code for the Stochastic Simulation project @ EPFL Lausanne.

## Code Structure :world_map:
- `parameters.py` contains a collection of test starting points; in order to run the other scripts, you need to choose the `testcase` at the beginning of the file (or, to try a different point, define the relevant parameters in this importable script).
- `numerical.py` contains the code for the numerical solution of the backwards parabolic PDE in `FeNiCS`.
- `simulation.py` contains the code to answer the stochastic part of the first question of the assignment. 
Select the task you want to perform between `twostagesMC` and `order` at the beginning of the script.
  - `twostagesMC` for the Two-Stages Monte Carlo algorithm to obtain stochastic estimates of the desired probability.
  - `order` to study the discretization order in time by running paths with different time refinements on the same Brownian paths.
- `variance_reduction.py`contains the implementation of the Antithetic Variables technique for variance reduction, and a comparison with Crude Monte Carlo.
- `splitting.py` contains the implementation of the splitting method; at the beginning of the script, you can choose how many circles use to divide the domain with the option `circles` and the task you want to perform between:
  - `FSextimation`, i.e. estimating the probability of the rare event with *Fixed Splitting* method.
  - `FSvariance`, i.e. studying the variance of the *Fixed Splitting* estimator. You can choose between two methods:
    - `Y1`, which computes the sample variance of the number of hits caused by the offspring of the various paths starting from (X_0,Y_0).
    - `R_m`, which repeats N = 100 times the algorithm and outputs the variance of the number of hits of the well. 
     
     :warning: *Be careful!* this option requires long computational time, and we advise you to disable printing before running it.   
  - `FEextimation`, i.e. estimating the probability of the rare event with *Fixed Effort* method.
- Folder `Plots` containing:
  - `plots_initial.py`, producing a plot of the starting points analyzed.
  - `order_plots.m`, `MATLAB` script producing a logarithmic plot of the error trend with respect to time discretization.
- Folder `Docs`, containing the assignment text, the sources exploited and the report.

## Author
👻 Giulia Mescolini [@giuliamesc](https://github.com/giuliamesc)
