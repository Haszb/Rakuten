from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.models import Variable
import json
from datetime import datetime, timedelta
import requests
import shutil
import logging
import shutil
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 25),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='A DAG for model retraining based on number of new product count and / or accuracy on new predictions',
    schedule_interval=timedelta(minutes=30),  
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
        print(f"Error obtaining JWT token: {response.status_code}, {response.text}")
        return None

def do_nothing():
    pass

def get_new_prod_data():
    file_path = '/app/data/new_product/new_prod_data.json'
    stats_url = f"{api_url}/Stats" 
    global token

    token = get_jwt_token(api_url, admin_username, admin_password)
    if token is None:
        print("Unable to obtain token, exiting...")
        return None

    headers = {"Authorization": f"Bearer {token}"}

    print("Requesting new_prod_data.json update...")
    response = requests.post(stats_url, headers=headers)  

    if response.status_code == 200:
        print("new_prod_data.json is now available.")
    else:
        print(f"Error during API request: {response.status_code}")
        return None
    
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        return data["Number of new products"], data["Calculated accuracy of new product (%)"]
    except Exception as e:
        print(f"Error reading new_prod_data.json: {e}")
        return None


def check_conditions(**kwargs):
    ti = kwargs['ti']

    prod_data = get_new_prod_data()

    number_of_new_products, new_products_accuracy = prod_data
    
    current_accuracy = float(Variable.get("current_accuracy", default_var=85))
    ti.xcom_push(key='prod_data', value=prod_data)
    
    if number_of_new_products > 500:
        if new_products_accuracy < current_accuracy:
            return 'backup'
        else:
            return 'do_nothing'
    else:
        if new_products_accuracy < current_accuracy:
            return 'check_difference'
        else:
            return 'do_nothing'

def check_difference(**kwargs):
    ti = kwargs['ti']

    prod_data = ti.xcom_pull(key='prod_data', task_ids='check_conditions')

    number_of_new_products, new_products_accuracy = prod_data
    current_accuracy = float(Variable.get("current_accuracy", default_var=85))
    
    if current_accuracy - new_products_accuracy >= 5:
        return 'backup'
    else: 
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
    pass

def retrain_model(**context):
    pass

def validation(**context):
    ti = context['ti']
    prod_data = ti.xcom_pull(key='prod_data', task_ids='check_conditions')
    number_of_new_products, new_products_accuracy = prod_data

    # Simulation
    new_trained_accuracy = 88.0 

    current_accuracy = float(Variable.get("current_accuracy", default_var=85))

    # Poussée des données pour utilisation future
    ti.xcom_push(key='number_of_products_processed', value=number_of_new_products)
    ti.xcom_push(key='model_accuracy_before', value=new_products_accuracy)
    ti.xcom_push(key='current_accuracy', value=current_accuracy)
    ti.xcom_push(key='new_trained_accuracy', value=new_trained_accuracy)

    is_model_better = new_trained_accuracy > new_products_accuracy

    if is_model_better:
        return 'email_success'
    else:
        return 'email_failure'

# Taches 
t0 = BranchPythonOperator(
    task_id='check_conditions',
    python_callable=check_conditions,
    dag=dag,
)

t0_1 = BranchPythonOperator(
    task_id='check_difference',
    python_callable=check_difference,
    dag=dag,
)

t1 = PythonOperator(
    task_id='backup',
    python_callable=backup,
    dag=dag,
)

t2 = PythonOperator(
    task_id='adjust_dataset',
    python_callable=adjust_dataset,
    dag=dag,
)

t3 = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
)

t4 = BranchPythonOperator(
    task_id='validation',
    python_callable=validation,
    dag=dag,
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
)

noop = PythonOperator(
    task_id='do_nothing',
    python_callable=do_nothing,
    dag=dag,
)

# Ordre
t0 >> [t0_1, t1, noop] 
t0_1 >> [t1, noop] 
t1 >> t2 >> t3 >> t4
t4 >> t5_a
t4 >> t5_b 
