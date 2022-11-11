#!/bin/sh
DIR="./Toxic"
mkdir $DIR
cd $DIR

rm -rf offenseval
wget --content-disposition https://cloud.tsinghua.edu.cn/f/ce557cadfe0343a7b849/?dl=1
unzip offenseval.zip
rm -rf offenseval.zip

rm -rf jigsaw
wget --content-disposition https://cloud.tsinghua.edu.cn/f/fe23ef717d374f1993b8/?dl=1
unzip jigsaw.zip
rm -rf jigsaw.zip

rm -rf twitter
wget --content-disposition https://cloud.tsinghua.edu.cn/f/32f576b3a61048ed8106/?dl=1
unzip twitter.zip
rm -rf twitter.zip

rm -rf hsol
wget --content-disposition https://cloud.tsinghua.edu.cn/f/26a8b1c7477d4c0897e2/?dl=1
unzip hsol.zip
rm -rf hsol.zip

cd ..
