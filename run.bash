#!/bin/bash

PYTHONPATH="/c/Users/Bei/anaconda3/python.exe"
echo "Starting server"
$PYTHONPATH server.py &
sleep 3  # Sleep for 3 seconds to give the server enough time to start

for i in $(seq 0 1); do
    echo "Starting client $i"
    $PYTHONPATH client.py --node-id ${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# Wait for all background processes to complete
wait

