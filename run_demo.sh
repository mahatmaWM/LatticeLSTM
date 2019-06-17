#!/usr/bin/env bash

python main.py --status train --train ./data/train.txt --dev ./data/dev.txt --test ./data/test.txt --savemodel ./data/test

#python main.py --status train --train ./data/train.txt --dev ./data/dev.txt --test ./data/test.txt --savemodel ./data/fm-play > ./music-play.log 2>&1 &