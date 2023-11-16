FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget sudo git make gcc g++ python3.11 python3.11-dev python3-pip\
    &&update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1\
    &&wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb\
    &&python3 -m pip install numpy h5py flask pandas graphistry\
    &&sudo dpkg -i cuda-keyring_1.1-1_all.deb\
    &&apt-get update && apt-get install -y cuda

WORKDIR /home/liy/packages
RUN git config --global user.name "yanggaoli"\
    &&git config --global user.email "yang98913745@163.com"\
    &&git clone https://github.com/pybind/pybind11.git

RUN mv /usr/local/cuda-12.* /home/liy/packages/cuda-12.2\
    &&rm -rf /usr/local/cuda*
ENV PATH="/home/liy/packages/cuda-12.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/home/liy/packages/cuda-12.2/lib64:${LD_LIBRARY_PATH}"

WORKDIR /home/liy/
RUN git clone https://github.com/yangaoli/a_elsa.git
RUN cp -r ./a_elsa/elsa ./&&rm -rf a_elsa

WORKDIR /home/liy/elsa/pybind_compcore/
RUN g++ -std=c++14 -fPIC -shared \
./*.cpp \
-I /usr/include/python3.11 \
-L /usr/lib/python3.11 \
-lpython3.11 \
-I/home/liy/packages/pybind11/include \
-O3 -o ../lsa/compcore.so

WORKDIR /home/liy/elsa/
RUN python3 -m pip install tqdm scipy
RUN python3 setup.py build&&python3 setup.py install
WORKDIR /home/liy/elsa/lsa/py_server/
RUN apt-get clean


CMD ["python3", "py_NET.py"]




# docker build -t elsa_server:2.0 .&&docker run -p 5002:5002 --gpus all -it elsa_server:2.0
# docker run --gpus all -it --name my-ubuntu-container elsa_server:1.0
# docker login
# docker tag elsa_server:1.0 bingwujiayin/elsa_server:1.0
# docker push bingwujiayin/elsa_server:1.0

