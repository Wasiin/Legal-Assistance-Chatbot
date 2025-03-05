FROM python:3.12
ENV PYTHONUNBUFFERED 1
RUN mkdir /bot-server
WORKDIR /bot-server
COPY pyproject.toml /bot-server/
RUN pipx install poetry
RUN pip install gunicorn
RUN poetry install
COPY . /bot-server/
WORKDIR /bot-server/src