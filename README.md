# antlr-deep-error-recovery

## How to Run

### Preparation
Install the required tools using the <code>prepare_environment.sh</code> script. It validates and/or installs Java, Python, and Maven. Moreover, it packages the jar, initialises the Python virtual environment, installs the required packages, and creates the data folders.

Next, make sure that you place the original data file in the following directory: <code>src/main/resources/data</code>

### Noise Generation
Now you can run the noisy data generation by executing the <code>generate_data.sh</code> script with the path of the original data and the noise percentage. Here is an example call: <code>./generate_data.sh src/main/resources/data/jhetas_clean.json 0.1</code>

The script returns the target directory where the noisy file is generated like so: <code>00_1</code>, which indicates the sub-directory for the given noise level. This sub-directory is needed for the next step, such that the next script knows where the noisy data was generated.

### Data Preparation & Model Training
Lastly, execute the <code>./train_model.sh</code> script and provide the following options: the directory of the noisy data, whether to load existing CV splits, and whether to resume training by loading an existing checkpoint. An example execution: <code>./train_model.sh 00_1 False False</code> if you want to load existing splits from the cache provide <code>True</code> as the second option and to load a checkpoint provide <code>True</code> as last parameter.
