#!/usr/bin/env bash

python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip3 install myokit  # Get Myokit, might need external installation for sundials
pip3 install git+https://github.com/pints-team/pints
