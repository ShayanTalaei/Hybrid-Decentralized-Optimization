#!/bin/bash

PROJECT_NAME="HDO"


run_sweep_and_agent () {
  # Set the SWEEP_NAME variable
  SWEEP_NAME="$1"

  # Run the wandb sweep command and store the output in a temporary file
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "hyperparams_opt/$SWEEP_NAME.yaml" >temp_output.txt 2>&1

  # Extract the sweep ID using awk
  SWEEP_ID=$(cat temp_output.txt | sed -n 's/.*wandb: Run sweep agent with: wandb agent \(.*\)/\1/p')
  
  # Remove the temporary output file
  rm temp_output.txt

  # Run the wandb agent command
  wandb agent --count 128 "$SWEEP_ID" --project "$PROJECT_NAME"
}

# list of sweeps to call
run_sweep_and_agent "cifar10_zeroth_sim_resnet"
