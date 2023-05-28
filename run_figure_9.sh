#!/bin/bash

# run command: nohup ./run_figure_9.sh >logs/run_figure_9.log &

python main_figure_9.py
cd draw_figure_9
python draw.py
cd ..