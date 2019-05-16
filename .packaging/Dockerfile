FROM python:3.7.0-slim-stretch
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./ /code
WORKDIR /code
RUN python3 setup.py develop

EXPOSE 8888

CMD jupyter notebook