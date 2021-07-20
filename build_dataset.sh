for lang in afb ail ame amh ara arz aym bra bul ces ckb ckt cni deu evn gup heb ind itl kmr, kod krl lud mag nld olo pol por rus sah see sjo spa syc tur tyv vep vro
do
python sigmorphon-2021/all_data/generate_data.py --src_dir=$1 --trg_dir=$3 --copy=True --lang=$lang
done

for lang in afb ail ame amh ara arz aym bra bul ces ckb ckt cni deu evn gup heb ind itl kmr, kod krl lud mag nld olo pol por rus sah see sjo spa syc tur tyv vep vro
do
	cat “${3}/${lang}.train” “${2}/${lang}.hall” > “${3}/${lang}.train”
	cp “${1}/${lang}.dev” “${3}/${lang}.dev”

done
python sigmorphon-2021/all_data/generate_data.py --src_dir=$3 --trg_dir=$3 --gen_all=True

