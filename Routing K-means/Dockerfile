FROM continuumio/miniconda3:latest

# Install the specific Python version
RUN conda install python=3.8

COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD ["python", "app.py"]