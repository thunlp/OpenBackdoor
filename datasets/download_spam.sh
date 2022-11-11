#!/bin/sh
DIR="./Spam"
mkdir $DIR
cd $DIR

rm -rf enron
wget --content-disposition https://cloud.tsinghua.edu.cn/f/7e1c72488a09450eb956/?dl=1
unzip enron.zip
rm -rf enron.zip

rm -rf lingspam
wget --content-disposition https://cloud.tsinghua.edu.cn/f/1d45eedea3954f068e45/?dl=1
unzip lingspam.zip
rm -rf lingspam.zip

cd ..
