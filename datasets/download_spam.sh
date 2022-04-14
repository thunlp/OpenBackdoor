#!/bin/sh
DIR="./Spam"
mkdir $DIR
cd $DIR

rm -rf enron
wget --content-disposition https://cloud.tsinghua.edu.cn/f/9e2d22a704f14ecf8828/?dl=1
unzip enron.zip
rm -rf enron.zip

rm -rf lingspam
wget --content-disposition https://cloud.tsinghua.edu.cn/f/50d5655e79624770bfe8/?dl=1
unzip lingspam.zip
rm -rf lingspam.zip

cd ..