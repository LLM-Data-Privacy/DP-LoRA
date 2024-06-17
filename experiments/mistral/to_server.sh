#!/bin/bash

local_file="experiments/4o_benchmark.py"
script_file="experiments/server_script.sh"
username="FNAIkqbe"
remote_host="blp04.ccni.rpi.edu"
remote_dir="kaiqi_bei"


scp "$local_file" ${username}@${remote_host}:${remote_dir}
scp "$script_file" ${username}@${remote_host}:${remote_dir}