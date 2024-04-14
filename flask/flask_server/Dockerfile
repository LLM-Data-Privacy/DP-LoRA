# Version: 1.0
FROM python:3-alpine3.12

WORKDIR /server

COPY . /server

RUN pip install -r requirements.txt

EXPOSE 3000

CMD python ./server.py