#! /usr/bin/env bash

echo "Working dir: "
pwd

echo "Fetching repo: "

git clone git@github.com:sadiuysal/Cmpe491-492.git
cd Cmpe491-492/
git checkout 492-dev-2


echo "Setting env: "

python3 -m venv env_492
source env_492/bin/activate

echo "Installing requirements: "
pip install -r requirements.txt



