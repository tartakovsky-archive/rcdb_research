FROM python:3.7.0
ENV PYTHONUNBUFFERED 1
ARG TALIB_TAR=ta-lib-0.4.0-src.tar.gz

RUN wget http://prdownloads.sourceforge.net/ta-lib/$TALIB_TAR && \
    tar -zxvf $TALIB_TAR && \
    rm -rf $TALIB_TAR && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    rm -rf ta-lib

WORKDIR /requirements
COPY ./requirements /requirements
COPY extra-index-url /requirements

RUN pip install numpy==1.16.4 && \
    pip install --extra-index-url $(cat extra-index-url) --no-cache-dir -r requirements.txt  -r requirements.dev.txt

COPY ./ /code
WORKDIR /code
RUN pip install -e .

EXPOSE 8888

CMD jupyter notebook