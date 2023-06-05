#!/bin/bash

# Read the value of num_scripts from Smith_Network_Script.py
num_scripts=$(grep -oP 'num_processes=\K\d+' Smith_Network_Script.py)

# Copy the base script rec_0.py to rec_$x.py
cp Smith_Network_Script.py rec_0_backup.py
sed -i 's/sol_linear_system=False/sol_linear_system=True/' Smith_Network_Script.py

# Create an array to store the background process IDs
declare -a pids

for ((x=0; x<num_scripts; x++)); do
    script="rec_3D_$x.py"
    cp rec_0_backup.py "$script"
    sed -i "s/process=0/process=$x/" "$script"
    sed -i "s/CheckLocalConservativenessFlowRate(/#CheckLocalCons/" "$script"
 
    python "$script" &  # Execute the script in the background
    pids[$x]=$!  # Store the process ID
done

# Wait for all background processes to finish
for pid in ${pids[@]}; do
    wait $pid
done

# Remove the backup script
rm rec_0_backup.py

for ((x=0; x<num_scripts; x++)); do
    script="rec_3D_$x.py"
    #rm "$script"
done
