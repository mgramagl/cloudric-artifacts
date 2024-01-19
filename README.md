# CloudRic Artifacts


## Data preparation

First, unzip the LPU_models data into the corresponding folder
```bash
unzip LPU_models.zip -d predictor_app_python/data/
```


## Software Dependencies
This software require some python packages, detailed in the file requirements.txt
You have two ways of running your code

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



