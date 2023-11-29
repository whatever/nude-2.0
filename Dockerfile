FROM python:3.11


RUN pip install --upgrade pip

WORKDIR /app

# pre-cook dependencies so they don't get re-ran everytime src gets modified
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY setup.cfg pyproject.toml .
COPY src src/
RUN pip install -e .

CMD ["python", "-m", "http.server", "8181"]
