FROM mcr.microsoft.com/devcontainers/cpp

RUN wget -qO- https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz | tar xz -C /opt 
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/onnxruntime-linux-x64-1.16.3/lib/"
