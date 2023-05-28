#!/bin/bash

# run command: nohup ./run_figure_10.sh >logs/run_figure_10.log &

python main_figure_10.py
cd draw_figure_10
python draw.py
cd ..