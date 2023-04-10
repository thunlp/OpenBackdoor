#!/bin/sh
DIR="./PlainText"
mkdir $DIR
cd $DIR

rm -rf webtext
wget --content-disposition https://cloud.tsinghua.edu.cn/f/275cde0a8ec54a5a87aa/?dl=1
unzip webtext.zip
rm -rf webtext.zip

cd ..

