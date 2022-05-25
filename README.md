# The Origins of Predictive Robustness in SVAEs

This is the repository associated with a paper currently in preparation and soon to be submitted. This paper analyzes the robustness properties observed in supervised variational autoencoders

## Model Descriptions

All models are implemented in the Code directory and subdirectories. These models all have associated tests in the Unit_Tests folder ensuring proper functioning. These models are all latent variable models and are implemented in the style of scikit-learn. Each model is an object and can be fit using the fit method. Transform methods and other relevant methods are also implemented in the model. Each file has documentation on what the models are in each file, as well as their purpose. Each method is also documented in the style of scikit-learn, with a sentence explaining its purpose, a list of all arguments and their associated type, as well as the outputs and their associated type. These models depend on functions defined in my Utils repository, however to make this code self-sufficient relevant functions have been copied over to this repository.

## Repository Components

Code - this directory has the model implementations

Demos - this directory has a series of examples, demonstrating how the code
can be used. This will be similar to the scripts in UnitTests, except it 
will be nicely formatted so that anyone can understand how to use these 
scripts with minimal python experience.

UnitTests - this directory has a list of unit tests to ensure proper 
functioning of the code. These are example scripts where the behavior is
known by default to ensure proper functioning of any code 
additions/modifications.

## Setup instructions 

Make sure anaconda is downloaded, with instructions found at 

https://www.anaconda.com/

Having installed Anaconda, you can create a conda environment to use the 
scripts. This can be done in terminal using the following commands:

```
conda create -n robustness python=3.10
conda activate robustness
pip install -r requirements_tensorflow.txt
```

Having installed the necessary packages you can now use this environment 
whenever you want to run the code. Type
```
conda activate robustness
```
to open this environment and type 
```
conda deactivate
```
once you have concluded your work. This will prevent you from accidentally
installing extra packages in the environment that can potentially break the 
implementation.

The important package for this work is Tensorflow 2.
