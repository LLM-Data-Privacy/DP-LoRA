# Version: 1.0
FROM python:3-alpine3.12
ENV PYTHONNUNBUFFERED  = 1

WORKDIR /client

COPY . /client

RUN pip install -r requirements.txt

EXPOSE 3000

CMD python -u ./client1.py