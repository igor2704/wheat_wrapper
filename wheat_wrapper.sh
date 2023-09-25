#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate igbus_env

if [[ "$#" -eq 3 ]]
then
	in_path="$1"
	out_path="$2"
    dir_count="$3"
else 
	echo "please, enter the path of the image directory"
	in_path=''
	read in_path
	
	echo "please, enter the path of the directory to write the result"
	out_path=''
	read out_path
    
    echo "please, enter the number of streams"
	dir_count=''
	read dir_count
fi

if [[ ! -e $in_path ]]
then
    echo "ERROR. This ${in_path} input path doesn't exist"
    exit
fi

if [[ ! -e $out_path ]]
then
    mkdir $out_path
fi

in_path=`realpath "${in_path}"`
out_path=`realpath "${out_path}"`

path=`pwd`
python3 create_working_dir.py "$in_path" "$dir_count" "$out_path"
working_path=`realpath "working_dir"`

python3 start_timer.py "${working_path}/global_timer.pkl"

let "i = 1"
while [[ $i -le $dir_count ]]
do
    if [[ ! -e "${out_path}/${i}" ]]
    then
        mkdir "${out_path}/${i}" &
    fi
    let "i++"
done
wait

let "i = 1"
while [[ $i -le $dir_count ]]
do

    if [[ ! -e "${out_path}/${i}/image_wheat_masks" ]]
    then
        mkdir "${out_path}/${i}/image_wheat_masks" &
    fi

    if [[ ! -e "${out_path}/${i}/detection" ]]
    then
        mkdir "${out_path}/${i}/detection" &
    fi
    
    if [[ ! -e "${out_path}/${i}/crops" ]]
    then
        mkdir "${out_path}/${i}/crops" &
    fi
    
    let "i++"
done
wait

echo ""
echo "############################"
echo "calculate segmentation masks"
echo "############################"
echo ""
python3 start_timer.py "${working_path}/segmentation_timer.pkl"
cd "${path}/segmentation"
let "i = 1"
while [[ $i -le $dir_count ]]
do
    ./infer -bone efficientnet-b2 -mn "model_efficientnet-b2.bin" --verbose --cuda -bs 32 -dp "${working_path}/${i}" -op "${out_path}/${i}/image_wheat_masks"
    let "i++"
done
wait

cd "$path"
python3 stop_timer.py "${working_path}/segmentation_timer.pkl" "segmentation time: "
sleep 2

python3 start_timer.py "${working_path}/detection_timer.pkl"
cd "${path}/detection"
echo ''
echo "#########################"
echo "calculate detection masks"
echo "#########################"
echo ''
let "i = 1"
while [[ $i -le $dir_count ]]
do
    python3 wrapper_lerok.py "${working_path}/${i}" "${out_path}/${i}/detection" "${out_path}/${i}/image_wheat_masks" "circles_2.pt" &
    let "i++"
done
wait
cd "$path"
python3 stop_timer.py "${working_path}/detection_timer.pkl" "detection time: "

let "i = 1"
while [[ $i -le $dir_count ]]
do
    python3 decrease_size_imgs.py "${out_path}/${i}/detection" &
    python3 copy_masks.py "${out_path}/${i}/image_wheat_masks" "${working_path}/${i}" &
    let "i++"
done
wait

let "i = 1"
while [[ $i -le $dir_count ]]
do
    python3 decrease_size_imgs.py "${out_path}/${i}/image_wheat_masks" &
    let "i++"
done
wait

python3 start_timer.py "${working_path}/pubescence.pkl"
cd "${path}/pubescence/Glume-pubescence-prediction-of-spikelets-main"
echo ''
echo "########################"
echo "predict glume pubescence"
echo "########################"
echo ''
let "i = 1"
while [[ $i -le $dir_count ]]
do
    python3 Nikita_wrapper.py "EfficientNet-B1.zip" "${working_path}/${i}" "${out_path}/${i}" &
    let "i++"
done
wait
cd "$path"
python3 stop_timer.py "${working_path}/pubescence.pkl" "predict glume pubescence time: "

python3 start_timer.py "${working_path}/extract_features_timer.pkl"
cd "${path}/feature_extracting"
echo ''
echo "################"
echo "extract features"
echo "################"
echo ''
let "i = 1"
while [[ $i -le $dir_count ]]
do
    java -Djava.library.path=lib \
    -jar werecognizer.jar "${working_path}/${i}" --out "${out_path}/${i}/features" --unet -d &
let "i++"
done
wait
    
cd "$path"
python3 stop_timer.py "${working_path}/extract_features_timer.pkl" "extracting features time: "
python3 start_timer.py "${working_path}/calculate_biomass.pkl"

echo ''
echo "#################"
echo "calculate_biomass"
echo "#################"
echo ''
let "i = 1"
while [[ $i -le $dir_count ]]
do
    python3 decrease_size_imgs.py "${out_path}/${i}/features" 425 425 &
    python3 create_crops.py "${working_path}/${i}" "${working_path}/${i}" "${out_path}/${i}/crops" &
    python3 calculate_biomass.py "${working_path}/${i}" "${out_path}/${i}/image_wheat_masks" &
    let "i++"
done
wait
python3 stop_timer.py "${working_path}/calculate_biomass.pkl" "calculating biomass time: "

cd "$path"
python3 glue_streams.py "$out_path" "$dir_count"
python3 merge_tables.py "$out_path"

python3 start_timer.py "${working_path}/search_outliers.pkl"
echo ''
echo "###############"
echo "search outliers"
echo "###############"
echo ''
python3 find_outliers.py "${out_path}/all_featurs.csv" "${out_path}/outliers_feauters.csv" "0.998"
python3 stop_timer.py "${working_path}/search_outliers.pkl" "searching outliers time: "

python3 stop_timer.py "${working_path}/global_timer.pkl" "total time: "

rm -r "$working_path"
