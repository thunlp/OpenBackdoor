#!/bin/sh
DIR="./SentimentAnalysis"
mkdir $DIR
cd $DIR

rm -rf imdb
wget --content-disposition https://cloud.tsinghua.edu.cn/f/37bd6cb978d342db87ed/?dl=1
tar -zxvf imdb.tar.gz
rm -rf imdb.tar.gz

cd ..
