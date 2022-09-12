#!/usr/bin/bash
#print user info
echo "$(id)"

# Define mlflow
export MLFLOW_TRACKING_URI=file:/project/mlruns
echo ${MLFLOW_TRACKING_URI}

# parse arguments
CMD=""
for i in $@; do
  if [[ $i == *"="* ]]; then
    ARG=${i//=/ }
    CMD=$CMD"--$ARG "
  else
    CMD=$CMD"$i "
  fi
done

# execute command
echo $CMD
$CMD
