
cd ./openbackdoor/attackers/poisoners/utils/style

if [ $1 = "bible" ]
then
  wget 'https://cloud.tsinghua.edu.cn/d/4fa2782123cc463384be/files/?p=%2Fbible.zip&dl=1'
  unzip 'index.html?p=%2Fbible.zip&dl=1' -d $1
  rm 'index.html?p=%2Fbible.zip&dl=1'
fi

if [ $1 = "shakespeare" ]
then
  wget 'https://cloud.tsinghua.edu.cn/d/4fa2782123cc463384be/files/?p=%2Fshakespeare.zip&dl=1'
  unzip 'index.html?p=%2Fshakespeare.zip&dl=1' -d $1
  rm 'index.html?p=%2Fshakespeare.zip&dl=1'
fi

if [ $1 = "poetry" ]
then
  echo 'https://cloud.tsinghua.edu.cn/d/4fa2782123cc463384be/files/?p=%2Fpoetry.zip&dl=1'
  unzip 'index.html?p=%2Fpoetry.zip&dl=1' -d $1
  rm 'index.html?p=%2Fpoetry.zip&dl=1'
fi

if [ $1 = "lyrics" ]
then
  wget 'https://cloud.tsinghua.edu.cn/d/4fa2782123cc463384be/files/?p=%2Flyrics.zip&dl=1'
  unzip 'index.html?p=%2Flyrics.zip&dl=1' -d $1
  rm 'index.html?p=%2Flyrics.zip&dl=1'
fi

if [ $1 = "tweets" ]
then
  wget 'https://cloud.tsinghua.edu.cn/d/4fa2782123cc463384be/files/?p=%2Ftweets.zip&dl=1'
  unzip 'index.html?p=%2Ftweets.zip&dl=1' -d $1
  rm 'index.html?p=%2Ftweets.zip&dl=1'
fi

cd ..