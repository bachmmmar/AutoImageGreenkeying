#!/bin/bash

SOURCEDIR=python/out14

if [ $@ -lt 1 ]; then
    mkdir -p small

    mogrify -quality 90 -path small -resize 1024x756 $SOURCEDIR/*.jpg
fi

ffmpeg -r 7 -pattern_type glob -i 'small/*.jpg' -c:v libx264 -crf 15 output.mp4
