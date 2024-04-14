# Version: 1.0
FROM python:3-alpine3.12

WORKDIR /client

COPY . /client

RUN pip install -r requirements.txt

EXPOSE 8000

CMD python ./client1.py