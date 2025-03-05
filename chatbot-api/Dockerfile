FROM python:3.12
ENV PYTHONUNBUFFERED 1
RUN mkdir /bot-server
WORKDIR /bot-server
COPY pyproject.toml /bot-server/
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install --no-root
RUN poetry add gunicorn
COPY . /bot-server/
WORKDIR /bot-server/src