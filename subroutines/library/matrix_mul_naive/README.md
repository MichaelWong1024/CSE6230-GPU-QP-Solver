This is a naive example for matrix multiplication on a CUDA GPU. Instructions:
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
    prompt% make
    ```

    The Makefile contains many useful bits of information on how to compile a CUDA code

3) Submit the example to Slurm using the sbatch command:

    ```
    prompt% sbatch cuda.sbatch
    ```

4) Compare the program output. 

```
diff batch.log batch.log.ref
diff myoutput.log myoutput.log.ref
```
