#!/bin/bash

SOURCEDIR=python/out/images

mkdir -p small

mogrify -quality 90 -path small -rotate 90 -resize 1024x756 $SOURCEDIR/*.jpg

ffmpeg -r 3 -pattern_type glob -i 'small/*.jpg' -c:v libx264 -crf 15 output.mp4
