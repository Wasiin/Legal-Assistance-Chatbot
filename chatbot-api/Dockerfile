FROM python:3.12
ENV PYTHONUNBUFFERED 1
RUN mkdir /bot-server
WORKDIR /bot-server
COPY pyproject.toml /bot-server/
RUN pip install -r pyproject.toml
RUN pip install gunicorn
COPY . /bot-server/
WORKDIR /bot-server/src