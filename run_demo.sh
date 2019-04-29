#!/usr/bin/env bash

python main.py --status train --train ./data/demo.train.char --dev ./data/demo.dev.char --test ./data/demo.test.char --savemodel ./data/demo > ./demo.log 2>&1 &

python main.py --status train --train ./data/train.txt --dev ./data/dev.txt --test ./data/test.txt --savemodel ./data/fm-play > ./music-play.log 2>&1 &

#python main.py --status decode --raw ../data/onto4ner.cn/demo.test.char --savedset ../data/onto4ner.cn/demo.dset --loadmodel ../data/onto4ner.cn/demo.0.model --output ../data/onto4ner.cn/demo.raw.out
