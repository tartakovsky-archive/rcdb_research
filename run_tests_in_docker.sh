#!/usr/bin/env bash
docker build -t research-notebooks--test -f .packaging/Dockerfile ./
docker run --rm research-notebooks--test:latest bash -c "flake8 . && pytest"
