#!/bin/bash

set -ev

cat << __END__ > requirements.txt
mecab-python3==0.996.5
bunkai==1.4.3
emoji==1.7.0
fugashi==1.1.2
sudachipy==0.6.6
sudachidict_core
ginza==5.1.2
ipadic==1.0.0
ja-ginza==5.1.2
janome==0.4.2
spacy
datasets==2.4.0
transformers==4.21.1
seqeval==1.2.2
small-text[transformers]==1.0.0
unidic-lite==1.0.8
cytoolz
__END__

pip install -r requirements.txt
