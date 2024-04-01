from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
import json
from datetime import datetime, timedelta
import requests
import logging
import shutil
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 31),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='A DAG for model retraining based on number of new product count and / or accuracy on new predictions',
    schedule_interval=timedelta(days=1),  
    doc_md="""
    ## Schedule Interval

    Runs daily.

    ## Default Arguments

    - **Owner:** `airflow`
    - **Start Date:** `March 31, 2024`
    - **Retries:** `3`
    - **Retry Delay:** `5 minutes`
    - **Catchup:** `False`

    ## Airflow Variables

    - `admin_username`: Username for API.
    - `admin_password`: Password for API.
    - `api_url`: API URL, defaults to `http://api:8000`.
    - `current_accuracy`: Current model accuracy, used for comparison with new accuracy (on new product and follow up training).

    ## Detailed Task Orchestration and Conditional Logic

    ### Task: `check_conditions`
    Decides the workflow path based on product data:

    - If `number_of_new_products` > 500:
    - Proceeds to `backup` if `new_products_accuracy` < `current_accuracy`.
    - Otherwise, does nothing.
    - If `number_of_new_products` ≤ 500:
    - Proceeds to `check_difference` if `new_products_accuracy` < `current_accuracy`.
    - Otherwise, does nothing.

    ### Task: `check_difference`
    Evaluates the difference in accuracy:

    - Proceeds to `backup` if the accuracy difference is ≥ 5.
    - Otherwise, does nothing.

    ### Task: `backup`
    Creates a backup of the current dataset before any major operation such as dataset adjustment or model retraining. This step ensures data integrity and provides a rollback point if needed.

    ### Task: `adjust_dataset`
    Prepares the dataset for retraining by adjusting or augmenting the new product data. This step is crucial for ensuring the model trains on relevant and up-to-date information.

    ### Task: `retrain_model`
    Initiates the retraining of the machine learning model with the adjusted dataset. This task aims to improve the model's accuracy with new data.

    ### Task: `validation`
    Validates the newly trained model against the existing model to ensure it meets the accuracy requirements. If the new model is better, it replaces the current model; otherwise, the system restores the previous model.

    ## Workflow Logic

    - The DAG starts with the `check_conditions` task to determine the necessary action based on the data's condition.
    - Depending on the outcome, it may either directly backup the data, check the difference in accuracy, or do nothing if conditions are not met.
    - Upon completing the `backup`, subsequent tasks prepare the dataset, retrain the model, and validate the new model's accuracy.
    - The final decision to update the model or revert to the old model is made based on the validation results.
    - Depending on the issue, a follow up e-mail with details is sent. 
    """
)

admin_username = Variable.get("admin_username")
admin_password = Variable.get("admin_password")
api_url = Variable.get("api_url", 'http://api:8000') 

def get_jwt_token(api_url: str, username: str, password: str):
    token_url = f"{api_url}/token"
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(token_url, data=data, headers=headers)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        logging.error(f"Error obtaining JWT token: {response.status_code}, {response.text}")
        return None

def do_nothing():
    pass

def get_new_prod_data(api_url, username, password):
    file_path = '/app/data/new_product/new_prod_data.json'
    stats_url = f"{api_url}/Stats"
    
    token = get_jwt_token(api_url, username, password)
    if token is None:
        logging.error("Unable to obtain token, exiting...")
        return None, None, None  
    
    headers = {"Authorization": f"Bearer {token}"}
    logging.info("Requesting new_prod_data.json update...")
    response = requests.post(stats_url, headers=headers)
    
    if response.status_code == 200:
        logging.info("new_prod_data.json is now available.")
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            return data.get('Number of new products'), data.get('Calculated accuracy of new product (%)'), token
        except Exception as e:
            logging.error(f"Error reading new_prod_data.json: {e}")
    else:
        logging.error(f"Error during API request: {response.status_code}")
    return None, None, None


def check_conditions(**kwargs):
    ti = kwargs['ti']
    
    admin_username = Variable.get("admin_username")
    admin_password = Variable.get("admin_password")
    api_url = Variable.get("api_url", 'http://api:8000')

    number_of_new_products, new_products_accuracy, token = get_new_prod_data(api_url, admin_username, admin_password)

    if number_of_new_products is None or new_products_accuracy is None or token is None:
        logging.error("Failed to get product data or token.")
        return 'do_nothing'
    
    current_accuracy = float(Variable.get("current_accuracy", default_var=85))
    ti.xcom_push(key='prod_data', value=(number_of_new_products, new_products_accuracy, token))
    
    logging.info(f"Number of new products: {number_of_new_products}, New products accuracy: {new_products_accuracy}, Current accuracy: {current_accuracy}")
    
    if number_of_new_products > 500:
        if new_products_accuracy < current_accuracy:
            logging.info("More than 500 new products with accuracy below the current accuracy, proceeding to backup.")
            return 'backup'
        else:
            logging.info("More than 500 new products but accuracy is not below the current accuracy, doing nothing.")
            return 'do_nothing'
    else:
        if new_products_accuracy < current_accuracy:
            logging.info("Less than or equal to 500 new products with accuracy below the current accuracy, checking difference.")
            return 'check_difference'
        else:
            logging.info("Less than or equal to 500 new products but accuracy is not below the current accuracy, doing nothing.")
            return 'do_nothing'

def check_difference(**kwargs):
    ti = kwargs['ti']
    
    prod_data = ti.xcom_pull(task_ids='check_conditions', key='prod_data')
    number_of_new_products, new_products_accuracy, _ = prod_data

    if number_of_new_products is None or new_products_accuracy is None:
        logging.error("Product data not available for comparison.")
        return 'do_nothing'

    current_accuracy = float(Variable.get("current_accuracy", default_var=85))
    
    logging.info(f"Checking accuracy difference... Current accuracy: {current_accuracy}, New products accuracy: {new_products_accuracy}")
    
    if current_accuracy - new_products_accuracy >= 5:
        logging.info("Accuracy difference is greater than or equal to 5, proceeding to backup.")
        return 'backup'
    else: 
        logging.info("Accuracy difference is less than 5, doing nothing.")
        return 'do_nothing'



def backup(**context):
    source_dir = Path('/app/data/new_product')
    datetime_now = datetime.now().strftime('%Y%m%d%H%M%S')
    destination_dir = Path(f'/app/data/archives/new_product/archive-{datetime_now}')
    destination_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy(item, destination_dir / item.name)

    logging.info(f"Backup completed successfully to {destination_dir}")

def adjust_dataset(**context):
    move_product_url = f"{api_url}/move_new_product"
    ti = context['ti']
    _, _, token = ti.xcom_pull(task_ids='check_conditions', key='prod_data')
    
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(move_product_url, headers=headers)

    if response.status_code == 200:
        logging.info("move_new_product API call successful.")
    else:
        logging.error(f"Error during move_new_product API call: {response.status_code}, {response.text}")


def retrain_model(**context):
    retrain_url = f"{api_url}/retrain"
    ti = context['ti']
    _, _, token = ti.xcom_pull(task_ids='check_conditions', key='prod_data')
    
    headers = {"Authorization": f"Bearer {token}"}

    logging.info("Starting model retraining process...")
    response = requests.post(retrain_url, headers=headers)

    if response.status_code == 200:
        logging.info("Model retraining successful.")
    else:
        logging.error(f"Error during model retraining API call: {response.status_code}, {response.text}")

def validation(**context):
    ti = context['ti']
    _, _, token = ti.xcom_pull(task_ids='check_conditions', key='prod_data')
    
    validation_url = f"{api_url}/validation"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(validation_url, headers=headers)

    if response.status_code != 200:
        logging.warning("Attempting to refresh token due to failed validation call...")
        token = get_jwt_token(api_url, admin_username, admin_password)
        if token is None:
            logging.error("Unable to obtain a new token, exiting...")
            return

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(validation_url, headers=headers)
        ti.xcom_push(key='token', value=token)

    if response.status_code == 200:
        new_trained_accuracy = response.json()
        ti.xcom_push(key='new_trained_accuracy', value=new_trained_accuracy)

        prod_data = ti.xcom_pull(key='prod_data', task_ids='check_conditions')
        number_of_new_products, new_products_accuracy, _ = prod_data
        current_accuracy = float(Variable.get("current_accuracy", default_var=85))

        logging.info(f"New trained accuracy: {new_trained_accuracy}, Current accuracy: {current_accuracy}, New products accuracy: {new_products_accuracy}")

        ti.xcom_push(key='number_of_products_processed', value=number_of_new_products)
        ti.xcom_push(key='model_accuracy_before', value=new_products_accuracy)
        ti.xcom_push(key='current_accuracy', value=current_accuracy)
               
        if new_trained_accuracy > current_accuracy or (new_trained_accuracy < current_accuracy and new_trained_accuracy > new_products_accuracy and new_trained_accuracy >= 0.9 * current_accuracy):
            Variable.set("current_accuracy", new_trained_accuracy)
            return 'email_success'
        else:
            old_model_dir = Path("./models/old")
            model_dir = Path("./models")
            if old_model_dir.exists():
                for file in old_model_dir.iterdir():
                    source = file
                    destination = model_dir / file.name
                    shutil.move(str(source), str(destination))
                logging.info("Model restoration completed due to validation failure.")
            else:
                logging.error("Model directory does not exist for restoration.")

            return 'email_failure'
    else:
        logging.error(f"Validation call failed after token refresh: {response.status_code}, {response.text}")
        return 'email_failure'

# Taches 
t0 = BranchPythonOperator(
    task_id='check_conditions',
    python_callable=check_conditions,
    dag=dag,
    doc_md="""
    ### `check_conditions`
    - **Type:** BranchPythonOperator
    - **Description:** Checks if the number of new products exceeds 500 and compares the accuracy of new products to the current accuracy to decide the next step: either proceed directly to backup, check the accuracy difference, or do nothing.
    """
)

t0_1 = BranchPythonOperator(
    task_id='check_difference',
    python_callable=check_difference,
    dag=dag,
    doc_md="""
    ### `check_difference`
    - **Type:** BranchPythonOperator
    - **Description:** Compares the accuracy difference between the current accuracy and that of the new products. If the difference is greater than or equal to 5, proceeds to backup; otherwise, does nothing.
    """
)

t1 = PythonOperator(
    task_id='backup',
    python_callable=backup,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED,
    doc_md="""
    ### `backup`
    - **Type:** PythonOperator
    - **Description:** Backs up new product data to an archive directory.
    """
)

t2 = PythonOperator(
    task_id='adjust_dataset',
    python_callable=adjust_dataset,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    doc_md = """
    ### `adjust_dataset`
    - **Type:** PythonOperator
    - **Description:** Adjusts the dataset by moving new products in preparation for retraining.
    """
)

t3 = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    doc_md="""
    ### `retrain_model`
    - **Type:** PythonOperator
    - **Description:** Initiates the model retraining process.
    """
)

t4 = BranchPythonOperator(
    task_id='validation',
    python_callable=validation,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    doc_md="""
    ### `validation`
    - **Type:** BranchPythonOperator
    - **Description:** Validates the model accuracy post-retraining. If accuracy is improved, sends a success email; otherwise, restores the old model and sends a failure email.
    """
)

t5_a = EmailOperator(
    task_id='email_success',
    to='gillesdeperetti.pro@gmail.com',
    subject='Model retraining success',
    html_content="""<h3>Model retraining succeeded</h3>
                    <p>Retraining information:</p>
                    <ul>
                        <li>Accuracy of the model in production (reference): {{ ti.xcom_pull(task_ids='validation', key='current_accuracy') }}%</li>
                        <li>Accuracy of the model on new product before retraining: {{ ti.xcom_pull(task_ids='validation', key='model_accuracy_before') }}%</li>
                        <li>Accuracy of new trained: {{ ti.xcom_pull(task_ids='validation', key='new_trained_accuracy') }}%</li>
                        <li>Number of new products processed: {{ ti.xcom_pull(task_ids='validation', key='number_of_products_processed') }}</li>
                    </ul>
                    <p>Please see the logs for more details.</p>""",
    dag=dag,
    doc_md = """
    ### `email_success`
    - **Type:** EmailOperator
    - **Description:** Sends a success email with retraining details if validation is successful.
    """
)

t5_b = EmailOperator(
    task_id='email_failure',
    to='gillesdeperetti.pro@gmail.com',
    subject='Model retraining failed',
    html_content="""<h3>Model retraining failed</h3>
                    <p>Model retraining failed during validation. Here are some details:</p>
                    <ul>
                        <li>Accuracy of the model in production (reference): {{ ti.xcom_pull(task_ids='validation', key='current_accuracy') }}%</li>
                        <li>Accuracy of the model on new product before retraining: {{ ti.xcom_pull(task_ids='validation', key='model_accuracy_before') }}%</li>
                        <li>Accuracy of new trained: {{ ti.xcom_pull(task_ids='validation', key='new_trained_accuracy') }}%</li>
                        <li>Number of new products processed: {{ ti.xcom_pull(task_ids='validation', key='number_of_products_processed') }}</li>
                    </ul>
                    <p>Please check the logs and correct any identified issues.</p>""",
    dag=dag,
    doc_md = """
    ### `email_failure`
    - **Type:** EmailOperator
    - **Description:** Sends a failure email with retraining details if validation fails.
    """
)

noop = PythonOperator(
    task_id='do_nothing',
    python_callable=do_nothing,
    dag=dag,
    doc_md = """
    ### `do_nothing`
    - **Type:** PythonOperator
    - **Description:** A task that performs no action. Used as a default path in branching operators.
    """
)

# Ordre
t0 >> [t0_1, t1, noop] 
t0_1 >> [t1, noop] 
t1 >> t2 >> t3 >> t4
t4 >> t5_a
t4 >> t5_b 
