#!/bin/bash

sudo apt update
sudo apt upgrade -y
sudo apt install build-essential zlib1g-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
sudo apt-get install -y libgl1-mesa-dev

wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
tar -xzvf Python-3.7.5.tgz
cd Python-3.7.5
./configure --prefix=/usr/local/src/python37
sudo make
sudo make install
sudo ln -s /usr/local/src/python37/bin/python3.7 /usr/bin/python3.7
sudo ln -s /usr/local/src/python37/bin/pip3.7 /usr/bin/pip3.7
cd ../

sudo apt-get install python-virtualenv
virtualenv -p python3.7 dualapp
source dualapp/bin/activate
pip install -r requirements.txt

python modify_file.py

