FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget sudo git make gcc g++ python3.11 python3.11-dev python3-pip\
    &&update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1\
    &&python3 -m pip install numpy h5py\
    &&wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb\
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
RUN git clone https://github.com/yangaoli/ac_elsa.git

WORKDIR /home/liy/elsa/pybind_compcore/
RUN make clean && make
WORKDIR /home/liy/elsa/lsa/py_html/
RUN apt-get clean

CMD ["python3", "py_NET.py"]

# docker build -t ubuntu/nvidia:1.0 .
# docker run --gpus all -it ubuntu/nvidia:1.0
# docker run --gpus all -it --name my-ubuntu-container ubuntu/nvidia:1.0
# docker login
# docker tag ubuntu/nvidia:1.0 bingwujiayin/ubuntu/nvidia:1.0
# docker push bingwujiayin/ubuntu/nvidia:1.0

