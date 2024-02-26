#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <sstream>
#include <filesystem>

#include "lpu_models.hpp"

using namespace std;

void load_trace(vector<vector<string>> *content)
{
  string fname = "../data/traces_236.8.csv";
  // vector<vector<string>> content;
  vector<string> row;
  string line, word;
  int count=0;
  fstream file (fname, ios::in);
  if(file.is_open())
  {
    getline(file, line);
    while(getline(file, line))
    {
        row.clear();

        stringstream str(line);

        while(getline(str, word, ' '))
        {
          row.push_back(word);
        }
        content->push_back(row);
        count++;
    }
  }
  else
  {
    printf("[ERROR] Could not open the file\n");
  }
}


void load_gtruth(map<tuple<float,int,int,int>,vector<float>> *content,string fname)
{
  // vector<vector<string>> content;
  tuple<float,int,int,int> key;
  float value;
  vector<float> values;
  string line, word;
  unsigned int fcount;
  fstream file (fname, ios::in);
  if(file.is_open())
  {
    getline(file, line);
    
    while(getline(file, line))
    {
        key = {0,0,0,0};
        fcount=0;
        stringstream str(line);

        while(getline(str, word, ','))
        {
          switch(fcount){
            case 0: //PRB
              std::get<3>(key) = std::stoi(word);
              break;
            case 1: //SNR
              std::get<0>(key) = std::stof(word);
              break;
            case 2: //MCS
              std::get<1>(key) = std::stoi(word);
              break;
            case 4:
              std::get<2>(key) = std::stof(word);
              break;
            case 17:
              value = std::stof(word);
          }

          fcount++;
        };
        auto it = content->find(key);
        if (it==content->end()){
          values = {value};
          content->insert({key,values});
        }else{
          it->second.push_back(value);
        }
    }
  }
  else
  {
    printf("[ERROR] Could not open the file\n");
  }
}

int main()
{
  // Load trace
  ofstream results;
  if (std::filesystem::is_directory("../results/cplus")) {
    results.open ("../results/cplus/results.csv");
  } else {
    results.open ("../../results/cplus/results.csv");
  }

  std::cout<<"[INFO] Loading data"<<std::endl;
  vector<vector<string>> content;
  load_trace(&content);
  std::map<std::tuple<float,int,int,int>,std::vector<float>> cpu_gtruth;
  std::map<std::tuple<float,int,int,int>,std::vector<float>> gpu_gtruth;
  load_gtruth(&cpu_gtruth,"../data/CPU_dataset.csv");
  load_gtruth(&gpu_gtruth,"../data/GPU_dataset.csv");

  // Create an instance of LPU Models
  std::string cpu_model_filepath = "../data/predictor_time_cpu.onnx";
  LpuModels *cpu_lpu_models = new LpuModels(cpu_model_filepath);
  std::string gpu_model_filepath = "../data/predictor_time_gpu.onnx";
  LpuModels *gpu_lpu_models = new LpuModels(gpu_model_filepath);
  
  // Run Inference
  std::vector<float> values(3);
  std::pair<std::vector<float>,float> cpu_res,gpu_res;
  std::vector<float> cpu_pred, gpu_pred;
  float cpu_time,gpu_time;

  results<<"SNR,MCS,PRBs,TBS,m_type,predicted_dec_time,dec_time,p_err,itime_latency"<<std::endl;
  std::tuple<float,int,int,int> key;
  std::set<std::tuple<float,int,int,int>> processed_input;

  std::cout<<"[INFO] Inference on data"<<std::endl;


  for(int i = 0; i < content.size(); ++i)
  {
    values = {stof(content[i][1])/MAX_SNR, stof(content[i][2])/MAX_MCS, stof(content[i][4])/MAX_TBS};
    cpu_res = cpu_lpu_models->inference(values);
    cpu_pred = std::get<0>(cpu_res);
    cpu_time = std::get<1>(cpu_res);

    gpu_res = gpu_lpu_models->inference(values);
    gpu_pred = std::get<0>(gpu_res);
    gpu_time = std::get<1>(gpu_res);


    key = {stof(content[i][1]),stoi(content[i][2]),stoi(content[i][4]),stoi(content[i][3])};

    auto already_it = processed_input.find(key);

    if(already_it==processed_input.end())
    {
      auto it = cpu_gtruth.find(key);  
      if (it!=cpu_gtruth.end()){
        for (auto & cpu_val : it->second) {
          float error = 100*((cpu_pred[0]-cpu_val)/cpu_val);
          results<<content[i][1]<<","<<content[i][2]<<","<<content[i][3]<<","<<content[i][4]<<",CPU,"<<cpu_pred[0]<<","<<cpu_val<<","<<error<<","<<cpu_time<<std::endl;
      
        }
      };
      
      auto it2 = gpu_gtruth.find(key);  
      if (it2!=gpu_gtruth.end()){
        for (auto & gpu_val : it2->second) {
          float error = 100*((gpu_pred[0]-gpu_val)/gpu_val);
          results<<content[i][1]<<","<<content[i][2]<<","<<content[i][3]<<","<<content[i][4]<<",GPU,"<<gpu_pred[0]<<","<<gpu_val<<","<<error<<","<<gpu_time<<std::endl;
      
        }
      }
      processed_input.insert(key);
    }; 
  }  
  results.close();

  std::cout<<"[INFO] Results generated in the result/cplus folder"<<std::endl;
}