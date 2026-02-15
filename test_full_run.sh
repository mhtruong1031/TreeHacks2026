#!/bin/bash
# Run full test and capture last 60 lines

python3 main_runtime.py --simulate test_data.csv --speed 0 --calibration-time 5 2>&1 | tail -60
