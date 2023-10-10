#!/bin/bash
path=$1
echo 'start postprocess'
mkdir $path/img
mkdir $path/annotations
mv $path/*.png $path/img
rm $path/_*.json
mv $path/*.json $path/annotations
mv $path/*.depth.exr $path/annotations
mv $path/*.seg.exr $path/annotations

