#!/bin/sh
DIR="./PlainText"
mkdir $DIR
cd $DIR

rm -rf webtext
wget --content-disposition https://cloud.tsinghua.edu.cn/f/9380814201c84243931c/?dl=1
unzip webtext.zip
rm -rf webtext.zip

cd ..

