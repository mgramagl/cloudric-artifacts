# CloudRic Artifacts

CloudRIC is a system that meets specific reliability targets in 5G FEC processing while sharing pools of heterogeneous processors among DUs, which leads to more cost- and energy-efficient vRANs. The details of the solution are presented in https://doi.org/10.1145/3636534.3649381. As described therein, CloudRIC exploits data-driven models of Logical Processing Units. This repository includes the LPU models of an Intel Xeon Gold 6240R CPU and of an NVIDIA GPU V100 and the tools to replicate the results shown in Figs. 16 and 17 of https://doi.org/10.1145/3636534.3649381. 

In addition to the C implementation used in our paper (for speed), we also provide a Python implementation for comparison. 

To use these models and replicate the results shown in Figs. 16 and 17 of https://doi.org/10.1145/3636534.3649381, we provide two methods: Docker (recommended) or Baremetal. 

## Docker method
### 1) Data preparation 

1.1) First, unzip the LPU_models data into the corresponding folder
```bash
unzip LPU_models.zip -d predictor_app_python/data/
unzip LPU_models.zip -d predictor_app_cplus/data/
```

This software require some python packages, detailed in the file requirements.txt
In addition, you need a working [Docker](https://www.docker.com/) and [docker-compose](https://docs.docker.com/compose/). This repository can also directly executed in Github codespaces and has all the needed software installed.

1.2) Launch the Docker containers

```bash
docker-compose up -d
```
### 2) PyTorch LPU model

2.1) Run the application

```bash
docker exec -w /app -it python python3 /app/app.py
```

2.2) The script will generate a CSV file in results/python folder. The file is a dataset with a number of test cases, each test case correspoding to a Transport Block with a different combinations of SNR, MCS and Radio Blocks. For each test case, the dataset includes the predicted processing time, the ground truth (measured offline), and the actual inference latency.

### 3) Onnx LPU model

3.1) (Optional) Convert the PyTorch model into Onnx. the Onnx model is already provided so this step is optional.

```bash
docker exec -w /app -it python python3 /app/conv.py
```

3.2) (Optional) The exported models are stored in the predictor_app_python/onnx_model folder. Copy them in the predictor_app_cplus/data folder.
```bash
cp predictor_app_python/onnx_model/*.onnx predictor_app_cplus/data/
```
3.3) Compile the C application that uses the model

```bash
docker exec -w /app -it cplus make
```

3.4. Run the application
```bash
docker exec -w /app/build -it cplus ./lpu_models_app
```

3.5) The application will generate a CSV file in results/cplus folder. The file is a dataset with the same cases tested used before using the ONNX model. 

### 4) Analyze the results

4.1. To analyze the results, we have prepared a script that creates plots visualizing the data. 

```bash
docker exec -w /app -it python python3 /app/results/plot.py
```
4.1. For each model (PyTorch or ONNX), there are two plots in the respective results/ folder.
- The file ecdf.pdf depicts the empirical CDF of the prediction error (WCET approach in Fig. 17 of XXX). Both PyTorch and ONNX models should provide the same performance.
- The file timings.pdf depicts the empirical CDF of the inference time. The ONNX model should provide substantially lower latencies, appropriate for real-time operation and presented in Fig. 16 of XXX for the AAL-B-CP, than the PyTorch model. The results may differ depending on the platform where the experiment is being run.

## Baremetal method

### 1) System preparation

You need a working installation of Python 3.11 with pip

1.1) Install the required packages

```bash
pip install --no-cache-dir -r requirements.txt
```

1.2) Unzip the LPU_models data into the corresponding folder
```bash
unzip LPU_models.zip -d predictor_app_python/data/
unzip LPU_models.zip -d predictor_app_cplus/data/
```

### 2) PyTorch LPU model

2.1) Run the script from the predictor_app_python folder
```bash
 cd predictor_app_python
 python3 app.py
```

2.2) The script will generate a CSV file in predictor_app_python/python folder. The file is a dataset with a number of test cases, each test case correspoding to a Transport Block with a different combinations of SNR, MCS and Radio Blocks. For each test case, the dataset includes the predicted processing time, the ground truth (measured offline), and the actual inference latency.


### 3) Onnx LPU model

3.1) (Optional) Convert the PyTorch model into Onnx. the Onnx model is already provided so this step is optional.

```bash
 cd predictor_app_python
 python3 conv.py
```

3.2) The exported models are stored in the predictor_app_python/onnx_model folder. Copy them in the predictor_app_cplus/data folder.

3.3) Download the ONNX runtime and uncompress it to the /opt folder
```bash
wget -qO- https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz | tar xz -C /opt 
```
3.4) Add the folder to the LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/onnxruntime-linux-x64-1.16.3/lib/
sudo ldconfig
```
3.5) Navigate to the source code folder and compile the application
```bash
cd predictor_app_cplus
make
```
3.6) Run the application
```bash
cd build
./lpu_models_app
```
3.7) The application will generate a CSV file in predictor_app_cplus/results folder. The file is a dataset with the same cases tested used before using the ONNX model. 


### 4) Analyze the results

4.1) Copy the corresponding CSV files in the results directory
```bash
mv predictor_app_python/results/results.csv results/python
mv predictor_app_cplus/results/results.csv results/cplus
```

4.2. To analyze the results, we have prepared a script that creates plots visualizing the data. 

```bash
 python3 results/plot.py
```
4.1. For each model (PyTorch or ONNX), there are two plots in the respective results/ folder.
- The file ecdf.pdf depicts the empirical CDF of the prediction error (WCET approach in Fig. 17 of XXX). Both PyTorch and ONNX models should provide the same performance.
- The file timings.pdf depicts the empirical CDF of the inference time. The ONNX model should provide substantially lower latencies, appropriate for real-time operation and presented in Fig. 16 of XXX for the AAL-B-CP, than the PyTorch model. The results may differ depending on the platform where the experiment is being run.

