FROM python:3.7.0
ENV PYTHONUNBUFFERED 1

COPY ./requirements /requirements

RUN pip install --no-cache-dir -r /requirements/requirements.txt  -r /requirements/requirements.dev.txt

COPY ./ /code
WORKDIR /code
RUN pip install -e .

EXPOSE 8888

CMD jupyter notebook