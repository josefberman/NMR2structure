# NMR2structure

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-1.6-orange)](https://xgboost.readthedocs.io/)

A machine learning pipeline to predict molecular substructures directly from 1D NMR spectra using gradient boosting.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

NMR2structure leverages gradient boosting classifiers to translate 1H and 13C NMR peak lists into 166 common molecular substructures (e.g., functional groups, ring systems). Trained on 6,569 compounds from the open-access NMRShiftDB2 database, our models decode subtle spectral signatures without manual feature engineering.

## Features

- Predicts 166 structural motifs from raw NMR peak lists  
- Specialized carbohydrate substructure model with AUC 0.93  
- General organic compounds model with AUC 0.77  
- Easy-to-use Python API for batch prediction and retraining  

## Prerequisites

- Python 3.7 or higher  
- [XGBoost](https://xgboost.readthedocs.io/)  
- scikit-learn  
- numpy  
- pandas  
- joblib  

## Installation

```bash
# Clone the repository
git clone https://github.com/josefberman/NMR2structure.git
cd NMR2structure

# Install dependencies
pip install xgboost scikit-learn numpy pandas joblib
```

## Data

The repository includes nmrshiftdb2withsignals.sd, an SD file containing 1D 1H and 13C NMR spectra (peak lists and chemical shifts) for 6,569 compounds from NMRShiftDB2.

## Usage

Below is a basic example of generating substructure predictions from your own NMR data:
```python
from database import load_spectrum_sd, maccs_to_substructures
from model import predict_bits_from_xgboost

# Load and preprocess a custom SD file
spectra_df = load_spectrum_sd("path/to/your_data.sd")

# Predict bit-vectors for structural motifs
bit_predictions = predict_bits_from_xgboost(spectra_df)

# Map bit-vectors to human-readable substructures
substructures = maccs_to_substructures(bit_predictions)
print(substructures.head())
```

To retrain or fine-tune models from scratch, use the main.py entry point:
```bash
python main.py --train --data nmrshiftdb2withsignals.sd --output models/
```

## Project Structure

```bash
NMR2structure/
├── main.py                       # Training and evaluation entry point
├── model.py                      # Gradient boosting model definitions
├── database.py                   # Data loading and substructure mapping
├── nmrshiftdb2withsignals.sd     # Raw NMRShiftDB2 dataset
└── README.md                     # Project documentation
```
## Model Training

* Extract peak lists and substructure labels from the SD file
* Optimize hyperparameters via Bayesian optimization
* Train XGBoost classifiers for each substructure
* Persist trained models in the models/ directory

## Evaluation

* General model (166 motifs): AUC = 0.77, Precision = 71%, Recall = 55%
* Carbohydrate model: AUC = 0.93, Precision = 93%, Recall = 88%

## Contributing

Contributions are welcome! To contribute:
* Fork the project
* Create your feature branch (git checkout -b feature/YourFeature)
* Commit your changes (git commit -m 'Add feature')
* Push to the branch (git push origin feature/YourFeature)
* Open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.

