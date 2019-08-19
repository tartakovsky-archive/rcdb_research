# /bin/bash

TALIB_TAR="ta-lib-0.4.0-src.tar.gz"
wget http://prdownloads.sourceforge.net/ta-lib/${TALIB_TAR} && \
    tar -zxvf ${TALIB_TAR} && \
    rm -rf ${TALIB_TAR} && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    rm -rf ta-lib
