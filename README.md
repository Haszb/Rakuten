Rakuten Product Classification Application
==============================

This project aims to develop and deploy a product classification application for Rakuten. It utilizes natural language processing to analyze textual product descriptions and computer vision to interpret product images, enabling the automated classification of products in Rakuten's catalog. This automation is designed to streamline the cataloging process, reducing the need for manual classification by product teams and improving efficiency.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── app                <- API for deploying models, making it easier for applications to use the predictions.
    │   ├── crud.py
    │   ├── database.py
    │   ├── db_models.py
    │   ├── main.py
    │   ├── schemas.py
    │   └── security.py
    ├── data
    │   ├── external       <- Data from third party sources, for predictions on external data.
    │   ├── preprocessed   <- The final, canonical data sets for modeling.
    │   │   ├── image_train <- Directory for train set images.
    │   │   ├── image_test  <- Directory for prediction set images.
    │   │   ├── X_train_update.csv <- CSV file with columns: designation, description, productid, imageid.
    │   │   ├── X_test_update.csv  <- CSV file for predictions with similar columns to X_train_update.csv.
    │   └── raw            <- The original, immutable data dump.
    │       ├── image_train <- Directory for train set images.
    │       ├── image_test  <- Directory for prediction set images.
    │
    ├── logs               <- Logs from training and predicting models.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks for initial data exploration and analysis.
    │
    ├── requirements.txt   <- Requirements file for reproducing the analysis environment.
    │
    ├── requirements_new.txt  <- Additional requirements for API functionality.
    │
    ├── src                <- Source code for this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   ├── main.py        <- Scripts to train models.
    │   ├── predict.py     <- Scripts to use trained models for making predictions.
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │   └── build_features.py
    │   │
    │   ├── models                
    │   │   └── train_model.py
    │   └── config         <- Parameters used in training and prediction scripts.
    │
    └── users.db           <- Database for user management in API.

Instructions for Setup and Execution
------------------------------------

1. Create and activate a Conda environment:
    - `conda create -n "Rakuten-project" python=3.9`
    - `conda activate Rakuten-project`

2. Install required packages:
    - `python3 -m pip install -r app/requirements.txt`

3. Import raw data:
    - `python3 src/data/import_raw_data.py`

4. Upload the image data folder set directly on local from the specified source, respecting the structure in `data/raw`.

        ├── data
        │   └── raw           
        |   |  ├── image_train 
        |   |  ├── image_test 

5. Prepare the dataset:
    - `python3 src/data/make_dataset.py data/preprocessed`

6. Train the models:
    - `python3 src/main.py`

7. Make predictions:
    - `python3 src/predict.py`
  
      Exemple : python3 src/predict1.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"

 The predictions are saved in data/preprocessed as 'predictions.json'

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>#cookiecutterdatascience</small></p>
