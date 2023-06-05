#!/bin/bash

# run command: nohup ./run_figure_3.sh >logs/run_figure_3.log &

python main_figure_3_approximation_domain.py
python main_figure_3_actual_domain.py
cd draw_figure_3
python draw.py
cd ..