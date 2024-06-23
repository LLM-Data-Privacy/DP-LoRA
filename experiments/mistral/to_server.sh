#!/bin/bash

local_file="mistral_benchmark_CD.py"
script_file="server_script.sh"
username="FNAIkqbe"
remote_host="blp04.ccni.rpi.edu"
remote_dir="barn/kaiqi_bei"


scp "$local_file" ${username}@${remote_host}:${remote_dir}
scp "$script_file" ${username}@${remote_host}:${remote_dir}