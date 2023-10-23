#! /usr/bin/env bash

while getopts f: option
do 
    case "${option}"
        in
        f)folder_path=${OPTARG};;
    esac
done

echo "folder_path:" $folder_path
readarray -d "/" -t strarr <<< "$folder_path"

test_model_template='compvis-word_@@@-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05'  # use @@@ to represent the class name
prompts_path="../data/${strarr[3]}.csv"
echo "prompts_path:" $prompts_path


for class in 'airplane' 'automobile' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'
do
    test_model_name=$(echo $test_model_template | sed "s/@@@/$class/")
    echo "$test_model_name"

    python classification.py \
        --folder "$folder_path/$test_model_name" \
        --output "$folder_path/annotation" \
        --prompt $prompts_path \
        --target_class $class
done


