#!/bin/bash

# run without neptune logging.
echo 'run the fine-tune...'
python cl_cf_keyword.py --epoch 10 >> log_finetune.txt

echo 'run the naive rehearsal...'
python cl_nr.py --epoch 10 >> log_nr.txt

echo 'run ewc...'
python cl_ewc.py --epoch 10  --elambda 5 >> log_ewc.txt

echo 'run si...'
python cl_si.py --epoch 10 --c 11 --elambda 0.0007 >> log_si.txt

echo 'run gem with buffer size: 128...'
python cl_gem.py --epoch 10 --bsize 128 >> log_gem_128.txt

echo 'run gem with buffer size: 512...'
python cl_gem.py --epoch 10 --bsize 512 >> log_gem_512.txt

echo 'run gem with buffer size: 1024...'
python cl_gem.py --epoch 10 --bsize 1024 >> log_gem_1024.txt

echo 'run tp-net...'
python cl_tpnet.py --epoch 10 >> log_tpnet.txt