#!/bin/bash

# run command: nohup ./run_demo.sh >logs/run_demo.log &

python demo.py
cd draw_table
python draw_demo_table.py
cd ..