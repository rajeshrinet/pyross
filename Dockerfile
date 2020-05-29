FROM continuumio/miniconda3:latest

WORKDIR /pyross

COPY . .

RUN apt update -y\
	&& apt install gcc g++ -y\
	&& conda env create --file environment.yml

SHELL ["conda", "run", "-n", "pyross", "/bin/bash", "-c"]

RUN python setup.py install \
	&& cp .githooks/pre-push .git/hooks/ \
	&& chmod +x .git/hooks/pre-push

ENTRYPOINT ["cd /home && conda activate pyross"]
