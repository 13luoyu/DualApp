#!/bin/bash

# run command: nohup ./run_table_2.sh >logs/run_table_2.log &

python main_table_2.py
cd draw_table
python draw_table_2.py
cd ..