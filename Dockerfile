FROM python:3.8-slim
RUN apt-get update -y \
    && apt-get -y --no-install-recommends install curl wget\
    && rm -rf /var/lib/apt/lists/* 
RUN pip3 install torch==1.9.0 \
                 torchvision==0.10.0 \
                 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app

# pre-cook dependencies so they don't get re-ran everytime src gets modified
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY setup.cfg pyproject.toml ./
COPY src src/
RUN pip3 install -e .
RUN pip3 install requests

RUN echo "start webserver here so we can put it on the nets"

CMD ["nude", "serve-api", "--checkpoint", "/checkpoints/nude2-dcgan-met-random-crop-198x198.pt", "--port", "8181"]
