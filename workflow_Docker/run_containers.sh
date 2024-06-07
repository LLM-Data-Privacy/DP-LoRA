#!/bin/bash


NETWORK=server_client
SERVER_IMAGE=rgtdthrd/flask-server
CLIENT_IMAGE=rgtdthrd/flask-client1
SERVER_CONTAINER_NAME=flask-server
CLIENT_CONTAINER_NAME=flask-client1
PORT=3000

echo "Creating Docker network..."
docker network create $NETWORK

echo "Building Docker images..."
docker build -t $SERVER_IMAGE ./flask_server
docker build -t $CLIENT_IMAGE ./flask_client1

echo "Running containers..."
docker run --name $SERVER_CONTAINER_NAME --network $NETWORK -p $PORT:$PORT -d $SERVER_IMAGE
docker run --name $CLIENT_CONTAINER_NAME --network $NETWORK -d $CLIENT_IMAGE

echo "All containers are running."
