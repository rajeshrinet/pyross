FROM continuumio/miniconda3:latest

WORKDIR /home

COPY . ./pyross

RUN apt update -y\
	&& apt install gcc g++ -y\
	&& conda env create --file pyross/environment.yml

SHELL ["conda", "run", "-n", "pyross", "/bin/bash", "-c"]

RUN python pyross/setup.py install \
	&& cp pyross/.githooks/pre-push pyross/.git/hooks/ \
	&& chmod +x pyross/.git/hooks/pre-push

ENV PATH /opt/conda/envs/pyross/bin:$PATH
