# antlr-deep-error-recovery

## How to Run

This section contains instructions on how to run the experiments.

### Preparation
Install the required tools using the <code>prepare_environment.sh</code> script. It validates and/or installs Java, Python, and Maven. Moreover, it packages the jar, initialises the Python virtual environment, installs the required packages, and creates the data folders.

Next, make sure that you place the original data file in the following directory: <code>src/main/resources/data</code>

### Running the whole Pipeline at once
Execute the <code>./run_benchmark.sh</code> script if you want to run the steps documented below at once. This script additionally zips and removes the data directories that we want to persist. Moreover, the script accepts multiple noise levels (in addition to the data file location which is the first parameter) to be executed in series. An example usage: <code>./run_benchmark src/main/resources/data/jhetas_clean.json 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8</code>. This command repeats the whole pipeline with noise levels 0.1, 0.2, ..., 12.8.

If you want to execute pipeline tasks manually, proceed with the following instructions.

### Noise Generation
To run the noisy data generation you may execute the <code>generate_data.sh</code> script with the path of the original data and the noise percentage. Here is an example call: <code>./generate_data.sh src/main/resources/data/jhetas_clean.json 0.1</code>

The script returns the target directory where the noisy file is generated like so: <code>00_1</code>, which indicates the sub-directory for the given noise level. This sub-directory is needed for the next step, such that the next script knows where the noisy data was generated.

### Data Preparation & Model Training
Lastly, execute the <code>./train_model.sh</code> script and provide the following options: the directory of the noisy data, whether to load existing CV splits, and whether to resume training by loading an existing checkpoint. An example execution: <code>./train_model.sh 00_1 False False</code> if you want to load existing splits from the cache provide <code>True</code> as the second option and to load a checkpoint provide <code>True</code> as last parameter.

### Generated Folders
To read the measurements and for reproducibility, the process creates the following folders:
```
.
└── src
    ├── main
    │   └── python
    │       └── data
    │       │   └── generated
    │       │       └── cache
    │       │       │   └── <NOISE_DIR>        # directory containing cross-validation folds in Pickle format (used for training)
    │       │       └── checkpoints
    │       │       │   └── <NOISE_DIR>        # directory containing PyTorch checkpoints
    │       │       └── cv
    │       │           └── <NOISE_DIR>        # directory containing cross-validation folds in JSON format
    │       └── logs
    │           └── <NOISE_DIR>                # directory containing log files, and accuracy/loss scores in JSON format    
    └── resources
        └── generated
            └── <NOISE_DIR>
                └── noisy_jhetas_clean.json    # the generated source file with added noise
```
Here ```<NOISE_DIR>``` denotes the target directory returned by the noise generation task.
