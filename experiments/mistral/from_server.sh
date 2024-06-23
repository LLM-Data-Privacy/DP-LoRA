#!/bin/bash

server_file="mistral_benchmark_RED.log"
script_file="server_script.sh"
username="FNAIkqbe"
remote_host="blp04.ccni.rpi.edu"
remote_dir="barn/kaiqi_bei/benchmark_log"


scp ${username}@${remote_host}:${remote_dir}/$server_file .
#scp "$script_file" ${username}@${remote_host}:${remote_dir}