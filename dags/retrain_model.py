from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.models import Variable
import json
from datetime import datetime, timedelta
import requests
import os

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 22),
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
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
    
    # Lecture du fichier mis à jour
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        return data["Number of new products"], data["Calculated accuracy of new product (%)"]
    except Exception as e:
        print(f"Error reading new_prod_data.json: {e}")
        return None





def check_conditions(**kwargs):
    # Récupérer les données actuelles 
    number_of_new_products, new_products_accuracy = get_new_prod_data()
    
    # Récupérer l'accuracy du modèle en production dans la variable Airflow
    current_accuracy = float(Variable.get("current_accuracy", default_var=85)) # au cas où elle n'y est pas on donne une valeur par défaut
    
    # Logique de décision
    if number_of_new_products > 500:
        if new_products_accuracy < current_accuracy:
            return 'backup'  # Condition A = True & B = True
        else:
            return 'do_nothing'  # Condition A = True & B = False, ne rien faire
    else:
        if new_products_accuracy < current_accuracy:
            return 'check_difference'  # Condition A = False & B = True
        else:
            return 'do_nothing' # Condition A = False & B = False, ne rien faire

def check_difference(**kwargs):
    _, new_products_accuracy = get_new_prod_data()
    
    current_accuracy = float(Variable.get("current_accuracy", default_var=85))
    
    # Vérifier si la différence justifie le réentrainement
    if current_accuracy - new_products_accuracy >= 5:
        return 'backup'
    else : 
        return 'do_nothing'

def backup(**context):
    pass

def adjust_dataset(**context):
    pass

def retrain_model(**context):
    pass

def validation(**context):
    is_model_better = True  

    if is_model_better:
        return 'email_success'
    else:
        return 'email_failure'

def email_success(**context):
    pass

def email_failure(**context):
    pass

def failure_closure(**context):
    pass

# Taches 
t0 = BranchPythonOperator(
    task_id='check_conditions',
    python_callable=check_conditions,
    provide_context=True,
    dag=dag,
)

t0_1 = BranchPythonOperator(
    task_id='check_difference',
    python_callable=check_difference,
    provide_context=True,
    dag=dag,
)

t1 = PythonOperator(
    task_id='backup',
    provide_context=True,
    python_callable=backup,
    dag=dag,
)

t2 = PythonOperator(
    task_id='adjust_dataset',
    provide_context=True,
    python_callable=adjust_dataset,
    dag=dag,
)

t3 = PythonOperator(
    task_id='retrain_model',
    provide_context=True,
    python_callable=retrain_model,
    dag=dag,
)

t4 = BranchPythonOperator(
    task_id='validation',
    python_callable=validation,
    provide_context=True,
    dag=dag,
)

t5_a = PythonOperator(
    task_id='email_success',
    provide_context=True,
    python_callable=email_success,
    dag=dag,
)

t5_b = PythonOperator(
    task_id='email_failure',
    provide_context=True,
    python_callable=email_failure,
    dag=dag,
)


t6 = PythonOperator(
    task_id='failure_closure',
    provide_context=True,
    python_callable=failure_closure,
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
t4 >> t5_b >> t6
