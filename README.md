# CloudRic Artifacts


## Data preparation

First, unzip the LPU_models data into the corresponding folder
```bash
unzip LPU_models.zip -d predictor_app_python/data/
unzip LPU_models.zip -d predictor_app_cplus/data/
```


## Software Components

We provide Pyhton and C code as used in our paper.

### Python prediction
This software require some python packages, detailed in the file requirements.txt
You have two ways of running the code

- $I$ Having  working installation of [Docker](https://www.docker.com/) and [docker-compose](https://docs.docker.com/compose/). This repository can be directly executed in Github codespaces and has all the needed software installed.
1. Launch the docker

```bash
docker-compose up -d
```
2. running the application

```bash
docker exec -w /app -it python python3 /app/app.py
```
3. The script will generate all the required files in the predictor_app_python/results folder



- $II$ Alternatively, you can directly install the software on a working installation of python3.11 with pip
1. Install the required packages

```bash
pip install --no-cache-dir -r requirements.txt
```
2. Run the script from the predictor_app_python folder
```bash
 cd predictor_app_python
 python3 app.py
```
3. The script will generate all the required files in the predictor_app_python/results folder


### (Optional) Conversting the PyTorch model into Onnx

Using the same configuration as the Python prediction, launch the converter
```bash
docker exec -w /app -it python python3 /app/conv.py
```

or

```bash
 cd predictor_app_python
 python3 conv.py
```

The exported models are stored in the predictor_app_python/onnx_model folder. If you want to use them with the C++ predictor discussed next, copy them in the predictor_app_cplus/data folder.



### C++ prediction

We also provide the C++ code to run the model in inference using ONNX, as we implemented in our evaluation.
Also in this case we provide a container for this purpose
1. Launch the docker

```bash
docker-compose up -d
```
2. compile the application

```bash
docker exec -w /app -it cplus make
```

3. run the application
```bash
docker exec -w /app/build -it cplus ./lpu_models_app
```

The generated files will be un the predictor_app_cplus/results folder

In case you dont want to use the provided containers you can
1. Download the ONNX runtime and uncompress it to the /opt folder
```bash
wget -qO- https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz | tar xz -C /opt 
```
2. Add the folder to the LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/onnxruntime-linux-x64-1.16.3/lib/
sudo ldconfig
```
3. Navigate to the source code folder and compile the application
```bash
cd predictor_app_cplus
make
```
4. run the application
```bash
cd build
./lpu_models_app
```
5. The script will generate the csv file in the predictor_app_cplus/results folder 

### Plots generation

1. (Optional) If the containers were not used, copy the corresponding csv files in the results directory
```bash
mv predictor_app_python/results/results.csv results/python
mv predictor_app_cplus/results/results.csv results/cplus
```
2. Run the plot generation script
```bash
docker exec -w /app -it python python3 /app/plot.py
```
3. Find the plots in the respective subfolders in the results folder.