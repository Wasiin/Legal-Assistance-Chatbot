FROM python:3.12
ENV PYTHONUNBUFFERED 1
RUN mkdir /bot-server
WORKDIR /bot-server
COPY requirements.txt /bot-server/
RUN pip install -r requirements.txt
RUN pip install gunicorn
COPY . /bot-server/
WORKDIR /bot-server/src