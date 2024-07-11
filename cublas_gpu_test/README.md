This is a simple example for vector addition on a CUDA GPU. Instructions:

1) load the cuda module
2) compile
3) submit the job to the Slurm workload management system to run on the cluster

### Basic usage:

1) Load the cuda module:

    ```
    prompt% module load cuda
    ```

2) Compile the example:

    ```
    prompt% make : will generate test case to compare multiplication 
     between cpu and gpu. GPU code use cublas.

    ```
    The Makefile contains many useful bits of information on how to compile a CUDA code

3) Submit the example to Slurm using the sbatch command:

    ```
    prompt% sbatch cuda.sbatch
    ```

4) Output is saved under myoutput.log
