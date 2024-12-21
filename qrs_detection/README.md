## QRS detection
This folder contains implementation of the ECG QRS detection, described in the `Ding_ecg.pdf` paper. 

The program used for detection of the files under qrs/ folder, containing detections is in the file `qrs_final.py` which uses utility functions from `utils.py`. 
There are additional files that show the work:
- `qrs.ipynb` shows the example on a single record
- `qrs_pipeline.py` was used for parameter grid search
- `gridsearch_results.csv` contains mean metrics values for all parameter value sets
- `plot_scores.ipynb` attempts to visualize performance with respect to a parameter value set and find the best one

To use the scripts you can set up an environment using the following commands:
- `pip venv` 
- `pip install -r requirements.txt`
