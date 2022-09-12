#!/bin/bash
export USER="${USER:=`whoami`}"
export HOME=/home/$USER

/usr/bin/env bash /neurodocker/startup.sh

#print user info
echo "`id`"

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

# execute comand
$CMD
