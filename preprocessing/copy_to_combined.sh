#!/bin/bash

# run from the tiles directory (e.g., /lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_tiles_single_sample_per_patient_13july/tiles/256px_9.9x)
# this will copy the tiles from their respective TCGA dirs to a dir named 'combined'
# uses gnuparallel 

# Step 1: Count the total number of files to transfer
total_files=$(find TCGA-* -type f | wc -l)
echo "Total files to transfer: $total_files"

# Get the current timestamp to calculate elapsed time later
start_time=$(date +%s)

# Step 2: Start the file transfer in the background
find TCGA-* -type f -print0 | parallel -0 -j 32 --progress cp {} combined/ &

# Get the PID of the parallel process
parallel_pid=$!

# Step 3: Monitor the progress
while kill -0 $parallel_pid 2>/dev/null; do
  transferred_files=$(find combined/ -type f | wc -l)
  percentage=$(echo "scale=2; $transferred_files / $total_files * 100" | bc)
  
  current_time=$(date +%s)
  elapsed_time=$(($current_time - $start_time))
  
  if [ $transferred_files -gt 0 ]; then
    estimated_total_time=$(echo "$elapsed_time / $transferred_files * $total_files" | bc -l)
    estimated_time_left=$(echo "$estimated_total_time - $elapsed_time" | bc -l)
  else
    estimated_time_left="Calculating..."
  fi
  
  elapsed_time_formatted=$(date -ud "@$elapsed_time" +%H:%M:%S)
  estimated_time_left_formatted=$(date -ud "@$estimated_time_left" +%H:%M:%S)
  
  echo "Files transferred so far: $transferred_files out of $total_files ($percentage%)"
  echo "Elapsed time: $elapsed_time_formatted"
  echo "Estimated time left: $estimated_time_left_formatted"
  
  sleep 10 # Check progress every 10 seconds
done

# Final count after the transfer is complete
transferred_files=$(find combined/ -type f | wc -l)
percentage=$(echo "scale=2; $transferred_files / $total_files * 100" | bc)
elapsed_time=$(($(date +%s) - $start_time))
elapsed_time_formatted=$(date -ud "@$elapsed_time" +%H:%M:%S)

echo "Final count - Files transferred: $transferred_files out of $total_files ($percentage%)"
echo "Total elapsed time: $elapsed_time_formatted"

