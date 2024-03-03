<h1>NMR_to_structure</h1>
This project demonstrates using machine learning to predict molecular substructures directly from NMR spectra.

<h2>Overview</h2>
The goal is to translate 1D NMR spectral data into structural motifs without relying on manual analysis or additional molecular information. This is achieved by training gradient boosting models on a dataset of 1H and 13C NMR spectra for over 6,500 compounds to decode signatures of functional groups and ring systems.

<h2>Key outcomes:</h2>

Models identify 166 structural fragments directly from NMR peaks<br>
Captures subtle environmental influences on resonance frequencies and intensities<br>
Achieves AUC 0.77, 71% precision, 55% recall predicting motifs on full dataset<br>
Carbohydrate model reaches AUC 0.93, 93% precision, 88% recall

<h2>Data</h2>
The dataset contains 1D 1H and 13C NMR spectra for 6,569 compounds from open-access database (NMRShiftDB2).<br>
Structural labels cover 166 common organic substructures like methyl groups, aromatic systems, etc.

<h2>Models</h2>
Gradient boosting models were implemented in XGBoost. Models take NMR peak lists as input and output predictions for the 166 structural motifs. Extensive hyperparameter optimization was performed using Bayesian optimization.

<h3>To apply the models to new NMR data:</h3>
Use the function <i>predict_bits_from_xgboost</i> from model.py and <i>maccs_to_substructures</i> from database.py<br> 

<h3>Contact</h3>
Assaf Berman (assaf.berman@ucdconnect.ie)
