FROM python:3.7.0
ENV PYTHONUNBUFFERED 1

COPY ./requirements /requirements
COPY extra-index-url /requirements

RUN pip install numpy==1.16.4 && \
    pip install --extra-index-url $(cat /requirements/extra-index-url) --no-cache-dir -r /requirements/requirements.txt  -r /requirements/requirements.dev.txt

COPY ./ /code
WORKDIR /code
RUN pip install -e .

EXPOSE 8888

CMD jupyter notebook