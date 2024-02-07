#!/bin/bash
# 注意：根据你的实际环境可能需要调整路径格式
# 如果是在 Git Bash 下，路径可能需要是类似于 /c/Users/Bei/anaconda3/python.exe 的格式
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

