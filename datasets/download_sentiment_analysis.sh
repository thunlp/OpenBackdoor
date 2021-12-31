#!/bin/sh
DIR="./SentimentAnalysis"
mkdir $DIR
cd $DIR

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
tar -zxvf imdb.tar.gz
rm -rf imdb.tar.gz

rm -rf SST-2
wget --content-disposition https://cloud.tsinghua.edu.cn/f/bccfdb243eca404f8bf3/?dl=1
tar -zxvf SST-2.tar.gz
rm -rf SST-2.tar.gz

cd ..
