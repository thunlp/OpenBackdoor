#!/bin/sh
DIR="./Toxic"
mkdir $DIR
cd $DIR

rm -rf offenseval
wget --content-disposition https://cloud.tsinghua.edu.cn/f/adebb6d5bde64223b8cc/?dl=1
unzip offenseval.zip
rm -rf offenseval.zip

rm -rf jigsaw
wget --content-disposition https://cloud.tsinghua.edu.cn/f/fe23ef717d374f1993b8/?dl=1
unzip jigsaw.zip
rm -rf jigsaw.zip

rm -rf twitter
wget --content-disposition https://cloud.tsinghua.edu.cn/f/a9ac2c756dee4608826b/?dl=1
unzip twitter.zip
rm -rf twitter.zip

cd ..