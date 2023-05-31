#!/bin/bash

# run command: nohup ./run_figure_8.sh >logs/run_figure_8.log &

python main_figure_8.py
cd draw_figure_8
python reorganize_data.py
./draw.sh
cd ..
cd draw_table
python draw_table_1.py
cd ..