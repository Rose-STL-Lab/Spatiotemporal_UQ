#!/bin/bash

week="week34"
mkdir ${week}result
for d in $(find ./$week/* -type d)
do
  #echo $d
  lastd=${d##*/}
  seednumber="${lastd//[^0-9]/}"
  echo $seednumber

  #lastf=$(find $d | sort -n -k1.4| tail -1)
  #filename=${lastf##*/}
  #echo $filename
  #suffix="${filename##*[0-9]}"
  #number="${filename%"$suffix"}"
  #eponumber="${number##*[!-0-9]}"

  eponumber="0"
  for f in $(find $d -type f)
  do
    filename="${f##*/}"
    suffix="${filename##*[0-9]}"
    number="${filename%"$suffix"}"
    epotmp="${number##*[!-0-9]}"
    if ((epotmp > eponumber));then
      eponumber=$epotmp
    fi
  done
  echo $eponumber
  sed -e "s,epochnumber,$eponumber,g" ./data/model/pretrained/COV-19/config.yaml>./data/model/pretrained/COV-19/config_tmp_0.yaml
  sed -e "s,weeknumber,$week,g" ./data/model/pretrained/COV-19/config_tmp_0.yaml>./data/model/pretrained/COV-19/config_tmp.yaml 
  config_filename=./data/model/pretrained/COV-19/config_tmp.yaml
  output_filename=./${week}result/seed${seednumber}epo${eponumber}.npz
  random_seed=$seednumber

  #echo $config_filename
  #echo $output_filename
  #echo $random_seed

  mkdir ${week}result/seed${seednumber}
  for (( c=0; c<20; c++ ))
  do
    output_filename=./${week}result/seed${seednumber}/${c}.npz
    python run_demo_pytorch.py --config_filename=$config_filename --output_filename=$output_filename --random_seed=$random_seed
  done
done

