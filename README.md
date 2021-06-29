# Cmpe492-sadiuysal
CMPE 492 Project Development Branch

## Updates
Please see wiki pages for weekly updates.

## Environmental setup and running:
Run below commands in order for training the model and saving trained models to outputs folder


### If you have setup.sh script and have access to repo then run :

* $ sh setup.sh
* $ cd Cmpe491-492
* $ sh run.sh

### Alternatively:

Create env: 

* $ python3 -m venv env_492
* $ source env_492/bin/activate

Upgrade pip:

* $ pip3 install --upgrade pip

Clone the repo and set current branch :

* $ git clone git@github.com:sadiuysal/Cmpe491-492.git
* $ cd Cmpe491-492/
* $ git checkout 492-dev-2

Install requirements and run:

* $ pip install -r requirements.txt

Change config.py if needed. Then:

* $ python3 run.py
* $ deactivate
