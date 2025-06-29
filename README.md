# Autonomous driver TORCS: Max Verstapte


## Used Libraries

```
- Tensorflow
- Keras
```

# How to use

## Pre-requisite
1. Anaconda python installed. In case anaconda is not installed, follow [Anaconda Installation](https://github.com/CognitiaAI/TORCS-Self-Driving-Agent) and install anaconda.
2. Clone the repository and import the environment by running `conda env create -f torcs_env.yml`.

## Run the server
1. torcs_server is the folder which is required for running the server. This server was downloaded from an open source github repository. The exe file named wtorcs.exe is the main server file which will be used to listen from client. Run wtorcs.exe file in torcs_server folder.
2. Click on Race.
3. Click on Race.
4. Click on New Race. The server will now wait for the client to connect 

## Run the client
1. Activate the enviornment i.e. `conda activate torcs_env`.
2. in the main folder start the client by running `python main.py`