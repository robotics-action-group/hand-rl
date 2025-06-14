#!/bin/sh

if [ "$1" = "sim" ]; then
   # docker compose only sim container
   docker compose stop sim
   docker compose rm -f sim
   docker compose run --service-ports --rm sim ${2:-}
   exit 0   
fi