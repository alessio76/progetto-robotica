#!/bin/bash

# Get the list of .png files in the "test" directory and store them in the vector
vector=($(find test -maxdepth 1 -type f -name "*.png"))

# Iterate over the elements of the vector
for element in "${vector[@]}"
do
    # Call dense_fusion_test.py with the --image parameter
    python3 dense_fusion_test.py --image "$element"
done

