#!/bin/sh

if [ "$1" = "sim-isaac" ]; then
   # docker compose only sim container
   docker compose stop sim-isaac
   docker compose rm -f sim-isaac
   docker compose run --service-ports --rm sim-isaac ${2:-}
   exit 0   
fi