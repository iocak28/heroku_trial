# **Bank Telecaller Decision Support System** 



## Introduction

This project is a Bank Telecaller Decision Support System. 
The main purpose is to help bank manager and telecaller quickly target customers who are more likely to subscribe to their product.

This project was pursued for fulfilment of the coursework ECE 229- Compuation data analysis and product development, offered by UC San Diego in Spring 2020. 

## Documentation

Full documentation can be seen at https://hazeltree.github.io/ECE229-Project/

## Table of Content

- [Methodology](#methodology)
- [Data source](#datasource)
- [Code](#code)
- [File Structure](#filestructure)
- [Require Packages](#requirepackages)
- [Contributors](#contributors)

---

## Methodology









----

## Data source

The UCI dataset can be found in the `./data/` directory. the`.txt` file is an introduction to the dataset.

The original data can be found at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing `./data/`

---

## Code

### Data Analysis

- pre_processing.py : Data loading and processing
- analysis.py : Analysis tools

### Feature Extraction
- feature_extraction.py : Feature Extractor
### Prediction
- prediction.py : Prediction on test data
---

## Directory Structure
```
├── dataset
|   ├── bank-additional-full.csv
|   ├── bank-additional-full.csv
|   └── bank-additional.csv
├── src
|    ├── pre_processing.py
|    ├── feature_extraction.py
|    └── prediction.py
├── doc
|    ├── build
|    |   ├── doctrees
|    |   |    ├── environment.pickle
|    |   |    └── index.doctree
|    |   └── html
|    |   |    ├── _sources
|    |   |    ├── _static
|    |   |    ├── .buildinfo
|    |   |    ├── genindex.html
|    |   |    ├── index.html
|    |   |    ├── objects.inv
|    |   |    ├── py-modindex.html
|    |   |    ├── search.html
|    |   |    └── searchindex.js
|    ├── source
|    |   |    ├── conf.py
|    |   |    └── index.rst
|    ├── Makefile
|    └── make.bat
├── test
|    ├── generate_coverage_report.py
|    ├── test_analysis.py
|    ├── test_dashboard.py
|    ├── test_feature_extraction.py
|    ├── test_pre_processing.py
|    ├── test_prediction.py
|    └── test_util.py
├── visualization
|    ├── analysis.py
|    ├── TODO
|    └── TODO
├── Dockerfile
├── LR_prediction.joblib
├── README.md
├── dashboard.py
├── docker-compose.yml
├── requirements.txt
├── temp-plot.html
└──  util.py
```
---

## Required Packages
- Python >=3.7
- dash
- dash-renderer
- dash-core-components
- dash-html-components
- dash-table
- plotly
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- xgboost
- scikit-learn
- coverage
- pytest
- joblib

Run pip install -r requirements.txt to set up your computer 

---

## Contributors
Chenhao Zhou,
Ismail Oack,
Xintong Zhou,
Amol Sakhale,
Harshita Krishna (h1krishn@ucsd.edu)
