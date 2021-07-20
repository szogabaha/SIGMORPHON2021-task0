for lang in afb ail ame amh ara arz aym bra bul ces ckb ckt cni deu evn gup heb ind itl kmr, kod krl lud mag nld olo pol por rus sah see sjo spa syc tur tyv vep vro
do
python sigmorphon-2021/inference.py exp_dir=$1 src_dir=$2 lang=$lang
python sigmorphon-2021/evaluate.py pred_dir=”${1}/inference” target_dir=$2 lang=$lang
done

