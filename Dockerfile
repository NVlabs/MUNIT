FROM nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04
# Set anaconda path
ENV ANACONDA /opt/anaconda
ENV PATH $ANACONDA/bin:$PATH
# Download anaconda and install it
RUN apt-get update && apt-get install -y wget build-essential
RUN apt-get update && apt-get install -y libopencv-dev python-opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev
RUN wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -P /tmp
RUN bash /tmp/Anaconda2-5.0.1-Linux-x86_64.sh -b -p $ANACONDA
RUN rm /tmp/Anaconda2-5.0.1-Linux-x86_64.sh -rf
RUN conda install -y pytorch torchvision cuda80 -c pytorch
RUN conda install -y -c anaconda pip 
RUN conda install -y -c menpo opencv
RUN conda install -y -c anaconda yaml
RUN pip install tensorboard



