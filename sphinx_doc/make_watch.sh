#!/bin/sh

while true;
do
  make "$@";
  echo "Waiting for changes..."
  inotifywait -qre close_write .; \
done
