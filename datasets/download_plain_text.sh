#!/bin/sh
DIR="./PlainText"
mkdir $DIR
cd $DIR

rm -rf webtext
wget --content-disposition https://cloud.tsinghua.edu.cn/f/382238ceb7174f36ad9e/?dl=1
unzip webtext.zip
rm -rf webtext.zip

cd ..

