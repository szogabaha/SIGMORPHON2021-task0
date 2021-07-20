for lang in afb ail ame amh ara arz aym bra bul ces ckb ckt cni deu evn gup heb ind itl kmr, kod krl lud mag nld olo pol por rus sah see sjo spa syc tur tyv vep vro
do
	python sigmorphon-2021/all_data/generate_data.py --src_dir=$1 --trg_dir=$2 --copy=True --lang=$lang
	cp “${1}/${lang}.dev” “${2}/${lang}.dev”
done

python sigmorphon-2021/all_data/generate_data.py --src_dir=$2 --trg_dir=$2 --gen_all=True

for lang in afb ail ame amh ara arz aym bra bul ces ckb ckt cni deu evn gup heb ind itl kmr, kod krl lud mag nld olo pol por rus sah see sjo spa syc tur tyv vep vro
do
	rm “${lang}.train”
	rm “${lang}.dev”
done

