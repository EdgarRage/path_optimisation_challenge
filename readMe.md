# Path Optimisation Project

## Overview

This project aims to optimize the construction of a beginner's trail in a mountainous region of the Free State, minimizing energy expenditure for participants. The solution includes ingestion of altitude map and energy expenditure data, modeling with machine learning techniques, A* search algorithm for path optimisation, and reporting the optimal path.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Data Files](#data-files)
- [Modelling and Optimisation](#modeling-and-optimisation)
- [Optimisation](#optimisation)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Requirements

In order to run the solution, you must know that the code was created and ran in Python 3.10.9 version.
Also, the packages used in the code are:
- numpy==1.23.5
- pandas==1.5.3
- matplotlib==3.7.0
- sckikit-learn==1.2.2

## Usage

The code to obtain the optimal path is: [path-optimisation-script-solution.py](path-optimisation-script-solution.py)

## Data Files

The available data for this problem is the following:
- **energy_cost.csv**
   - This table contains the measurements from a sports science lab that relates the walking gradient/slope on a treadmill with the energy expended of multiple test subjects. 
     > *Energy expenditure is measured in $J.kg^{-1}.min^{-1}$, calculated based on oxygen consumption and treadmill speed*
- **altitude_map.csv**
   - This table contains information that describes an altitud map of the region of interest.
     > *The altitude map is at resolution of 10m x 10m, measurements are in meters, with North and the Y-axis going up vertically*

For more details of the process, see the [path-optimisation-nb-solution.ipynb](path-optimisation-nb-solution.ipynb) file in the repository.

## Modeling and Optimisation

For more details of the process, see the [path-optimisation-nb-solution.ipynb](path-optimisation-nb-solution.ipynb) file in the repository.


## Results

For more details, see the [optimal_path_coords.csv](optimal_path_coords.csv) and [optimal_path_visualization.png](optimal_path_visualization.png) files in the repository.

## Future Improvements

For more details, see the [optimal_path_advice.txt](optimal_path_advice.txt) file in the repository.
