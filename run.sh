#! /usr/bin/env bash

echo "Working dir: "
pwd

echo "Running the model: "
python3 run.py

echo "Committing outputs to git: "
git commit -a -m "Model outputs. "

echo "Pushing outputs to git: "
git push


