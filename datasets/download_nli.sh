#!/bin/sh
DIR="./NLI"
mkdir $DIR
cd $DIR

rm -rf mnli
wget --content-disposition https://cloud.tsinghua.edu.cn/f/33182c22cb594e88b49b/?dl=1
tar -zxvf mnli.tar.gz
rm -rf mnli.tar.gz

cd ..