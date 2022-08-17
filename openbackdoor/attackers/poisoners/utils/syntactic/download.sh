
cd ./data

mkdir TProcess.NLTKSentTokenizer
cd TProcess.NLTKSentTokenizer
wget https://data.thunlp.org/TAADToolbox/punkt.english.pickle.zip --no-check-certificate
unzip ./punkt.english.pickle.zip
rm -r ./punkt.english.pickle.zip
cd ..

mkdir TProcess.NLTKPerceptronPosTagger
cd TProcess.NLTKPerceptronPosTagger
wget https://data.thunlp.org/TAADToolbox/averaged_perceptron_tagger.pickle.zip --no-check-certificate
unzip ./averaged_perceptron_tagger.pickle.zip
rm -r ./averaged_perceptron_tagger.pickle.zip
cd ..

mkdir TProcess.StanfordParser
cd TProcess.StanfordParser
wget https://data.thunlp.org/TAADToolbox/stanford_parser_small.zip --no-check-certificate
unzip ./stanford_parser_small.zip
rm -r ./stanford_parser_small.zip
cd ..

mkdir AttackAssist.SCPN
cd AttackAssist.SCPN
wget https://data.thunlp.org/TAADToolbox/scpn.zip --no-check-certificate
unzip ./scpn.zip
rm -r ./scpn.zip
cd ..

cd ..