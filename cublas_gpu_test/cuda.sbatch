#!/bin/bash

# Request 2 nodes, 1 GPU, 32 GB memory.
#SBATCH -N2 --mem-per-cpu=16G --gres=gpu:1

# Request 10 minutes of wall-clock time.
#SBATCH -t10

# See https://gatech.service-now.com/technology?id=kb_article_view&sysparm_article=KB0042111 for more information on how to set resources

# Set the output log to be batch.log
#SBATCH -obatch.log

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Load the cuda module so we can find the run-time libraries
module load cuda

# rm myoutput.log

# Run the code
# ./matmul 256 512 1024 > myoutput.log
# ./matmul 512 1024 2048 >> myoutput.log
# ./matmul 1024 2048 4096 >> myoutput.log
# # ./matmul 2049 4100 5000 >> myoutput.log

./matmul 5 6 5 > myoutput.log
./matmul 32 32 32 >> myoutput.log
./matmul 64 64 64 >> myoutput.log
./matmul 128 128 128 >> myoutput.log
./matmul 256 256 256 >> myoutput.log
./matmul 512 512 512 >> myoutput.log
./matmul 1024 1024 1024 >> myoutput.log
./matmul 2048 2048 2048 >> myoutput.log

# ./matmul > myoutput.log
