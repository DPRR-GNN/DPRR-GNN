FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV PYTHON_VERSION 3.9.9
RUN mkdir /code
WORKDIR /code
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    git \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    tzdata \
&& git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
&& $PYENV_ROOT/plugins/python-build/install.sh \
&& /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
&& rm -rf $PYENV_ROOT

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.in-project true

ADD ./pyproject.toml /code/pyproject.toml
RUN poetry lock
RUN poetry install --no-dev
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html


