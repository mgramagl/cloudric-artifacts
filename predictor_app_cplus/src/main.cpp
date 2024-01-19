#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "lpu_models.hpp"

using namespace std;

void load_trace(vector<vector<string>> *content)
{
  string fname = "../data/traces_236.8.csv";
  // vector<vector<string>> content;
  vector<string> row;
  string line, word;

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
  vector<vector<string>> content;
  load_trace(&content);

  // TODO: load ground truth (CPU_dataset and GPU_dataset)

  // Create an instance of LPU Models
  std::string cpu_model_filepath = "../data/predictor_time_cpu.onnx";
  LpuModels *cpu_lpu_models = new LpuModels(cpu_model_filepath);
  std::string gpu_model_filepath = "../data/predictor_time_gpu.onnx";
  LpuModels *gpu_lpu_models = new LpuModels(gpu_model_filepath);

  // Run Inference
  std::vector<float> values(3);
  std::vector<float> cpu_pred, gpu_pred;
  for(int i = 0; i < content.size(); ++i)
  {
    printf("SNR: %f MCS: %f TBS: %f\n", stof(content[i][1]), stof(content[i][2]), stof(content[i][4]));
    values = {stof(content[i][1])/MAX_SNR, stof(content[i][2])/MAX_MCS, stof(content[i][4])/MAX_TBS};
    cpu_pred = cpu_lpu_models->inference(values);
    gpu_pred = gpu_lpu_models->inference(values);
    printf("pred_cpu: %f pred_gpu:%f\n", float(cpu_pred[0]), float(gpu_pred[0]));
    
    // TODO: compute error

    // TODO: Store results into file. Header = SNR,MCS,PRBs,TBS,m_type,predicted_dec_time,dec_time,p_err

    if (i == 5)
      break;
  }  
}