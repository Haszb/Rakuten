#!/bin/bash

export $(grep -v '^#' .env | xargs)

cat << EOF > ./flow/airflow_variables.json
{
    "admin_password": "$ADMIN_PASSWORD",
    "admin_username": "$ADMIN_USERNAME",
    "api_url": "http://api:8000",
    "current_accuracy": "71.0"
}
EOF
