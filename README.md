Income Prediction
-------------------------

Predict whether a person's income is above 50K or below based on various factors from the censor data. 

Installation
-------------------------
### Install the requirements 
* It is recommended to use anaconda virtual environment to run the notebook. Please run `conda create -n income_prediction_env python=3.7.3 anaconda` after anaconda distribution has been successfully downloaded and installed. 
* Install additional required packages using `pip install -r requirements.txt`.


Run the code
--------------------------
* Run `python model_training_evaluate.py` to start preprocessing of the data and the subsequent algorithm evaluation through nested cross validation. 
* After nested cross validation is done, you will be asked to choose the best algorithm for testing purpose according to the results returned by prior cross validation method. 

Reference
--------------------------
* https://scikit-optimize.github.io/ for documentation on using Bayesian method to do model selection on Python
* https://towardsdatascience.com/a-short-introduction-to-model-selection-bb1bb9c73376 for an introduction to nested cross validation for algorithm and hyperparameter selection
* https://www.kaggle.com/kashnitsky/mlcourse for the dataset





