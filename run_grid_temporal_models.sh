#!/bin/bash

# Define the grid of argument values
dataset_values=("duffing" "vanderpol")
model_type_values=("TCN" "GRU" "LSTM")
lifted_state_values=(4 8 16)
time_window_values=(4 8 16)

# # Flag to skip the first combination
# skip_first=true

# Loop through each combination of arguments
for dataset in "${dataset_values[@]}"; do
  for model_type in "${model_type_values[@]}"; do
    for lifted_state in "${lifted_state_values[@]}"; do
      for time_window in "${time_window_values[@]}"; do
        # if $skip_first; then
        #   skip_first=false
        #   continue
        # fi
        echo "Running script.py with dataset=$dataset, model_type=$model_type, lifted_state=$lifted_state, time_window=$time_window"
        python3 train_model.py -d "$dataset" -m "$model_type" -ls "$lifted_state" -tw "$time_window"
      done
    done
  done
done