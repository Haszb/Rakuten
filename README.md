Rakuten Product Classification Application
==============================

This project aims to develop and deploy a product classification application for Rakuten. It utilizes natural language processing to analyze textual product descriptions and computer vision to interpret product images, enabling the automated classification of products in Rakuten's catalog. This automation is designed to streamline the cataloging process, reducing the need for manual classification by product teams and improving efficiency.

Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── app                         <- API folder
    │   ├── Dockerfile              <- Dockerfile for building the Docker image of the API.
    │   ├── Tests                   <- Directory for API unit tests.
    │   │   └── test_api.py         <- Test file for the API.
    │   ├── __init__.py
    │   ├── crud.py
    │   ├── database.py
    │   ├── db_models.py
    │   ├── main.py                 <- API routes and logic.
    │   ├── requirements.txt        <- Dependencies file for the API environment.
    │   ├── schemas.py
    │   └── security.py
    ├── docker-compose.yml         <- Docker Compose configuration file for deploying the application.
    ├── flow
    │   ├── dags                    <- Directory for Airflow DAG definition files.
    │   │   └── retrain_model.py    <- Script for the Airflow retrain model DAG.
    │   ├── logs                    <- Directory for Airflow logs.
    │   └── plugins                 <- Directory for custom Airflow plugins.
    ├── init_airflow_variables.sh  <- Script to initialize environment variables for Airflow.
    ├── logs                        <- Directory for training logs.
    ├── models                      <- Directory for trained and serialized models, model predictions, or model summaries.
    │   ├── old                     <- Directory for old versions of models.
    ├── notebooks                   <- Jupyter notebooks for initial data exploration and analysis.
    │   └── Rakuten.ipynb          <- Jupyter Notebook for Rakuten project analysis.
    ├── setup.py                    <- Configuration file for package installation.
    └── src                         <- Source code for this project.
        ├── __init__.py
        ├── config                  <- Configuration files for the project.
        ├── docs_files              <- Documentation files for the project.
        │   └── workflow.png        <- Image of the Airflow workflow.
        ├── features                <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py   <- Script for building features.
        ├── main.py                 <- Scripts to train models.
        ├── models                  <- Scripts to train models.
        │   ├── __init__.py
        │   └── train_model.py      <- Script for training models.
        ├── predict.py              <- Scripts to use trained models for making predictions.
        └── visualization           <- Scripts for data visualization and model results visualization.
            ├── __init__.py
            └── visualize.py       <- Script for data visualization.


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
  
      Example: `python3 src/predict.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"`

 The predictions are saved in data/preprocessed as 'predictions.json'


Prerequisite: creation of .env 
------------------------------------

To run the application, you need to create a `.env` file at the root of the project with the following environment variables:

    # API
    SECRET_KEY=your_secret_key
    DATABASE_URL=sqlite:///./users.db
    ADMIN_USERNAME=admin
    ADMIN_EMAIL=admin@admin.com
    ADMIN_PASSWORD=your_admin_password

    # Airflow
    AIRFLOW__CORE__EXECUTOR=LocalExecutor
    AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:your_airflow_password@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY=your_fernet_key
    AIRFLOW__CORE__LOAD_EXAMPLES=false
    _AIRFLOW_DB_UPGRADE=true
    _AIRFLOW_WWW_USER_CREATE=true
    _AIRFLOW_WWW_USER_USERNAME=admin
    _AIRFLOW_WWW_USER_PASSWORD=your_airflow_admin_password

    # SMTP pour Airflow
    AIRFLOW__EMAIL__EMAIL_BACKEND=airflow.utils.email.send_email_smtp
    AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
    AIRFLOW__SMTP__SMTP_STARTTLS=True
    AIRFLOW__SMTP__SMTP_SSL=False
    AIRFLOW__SMTP__SMTP_USER=your_email@gmail.com
    AIRFLOW__SMTP__SMTP_PASSWORD=your_smtp_password
    AIRFLOW__SMTP__SMTP_PORT=587
    AIRFLOW__SMTP__SMTP_MAIL_FROM=your_email@gmail.com

    # Postgres - DB AirFlow
    POSTGRES_USER=airflow
    POSTGRES_PASSWORD=your_postgres_password
    POSTGRES_DB=airflow

Docker Containers Setup and Execution
------------------------------------

1. Ensure Docker and Docker Compose are installed on your system.

2. Use the provided script to set up Airflow variables:
    - Run `chmod +x init_airflow_variables.sh` to make the script executable.
    - Execute the script `./init_airflow_variables.sh` to generate `flow/airflow_variables.json`.

3. Build and start the Docker containers with Docker Compose:
    - Execute `docker-compose up --build` to build and start the containers.

API Documentation
-----------------

Once the containers are running, the API is accessible at `http://0.0.0.0:8000/`. 

- **Swagger UI**: Access the interactive API documentation at `http://0.0.0.0:8000/docs`. Swagger UI allows you to directly execute API calls from the browser.
  
- **ReDoc**: For a different documentation format, visit `http://0.0.0.0:8000/redoc`, which also provides a clear and organized interface for the API's endpoints and models.
  
- **OpenAPI Specification**: The OpenAPI specification file can be accessed directly at `http://0.0.0.0:8000/openapi.json`, providing a standard, language-agnostic interface to RESTful APIs.

Airflow Documentation
---------------------

The Airflow interface can be accessed at http://0.0.0.0:8080/. You can access the workflow dedicated to retraining, logs and trigger tasks. 

- The retraining workflow is organized as follows: 

![Workflow AirFlow](src/docs_files/workflow.png)

- Workflow documentation is available in the `DAG Docs` drop-down menu when you consult the DAG: model_retraining from the web interface. 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>#cookiecutterdatascience</small></p>
