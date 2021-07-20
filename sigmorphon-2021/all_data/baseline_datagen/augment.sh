#!/bin/bash
for lng in afb ame ara aym bul ckb cni evn heb itl kod lud nld pol rus see syc vep ail amh arz bra ces ckt deu gup ind kmr krl mag olo por sah spa tyv
do
    echo "Augmenting language " ${lng}
    python augment.py part1/original $lng --examples 10000
done
