#!/bin/bash

# run without neptune logging.
echo 'run the fine-tune...'
python cl_cf_keyword.py --epoch 6 >> log_finetune.txt

echo 'run the naive rehearsal...'
python cl_nr.py --epoch 6 >> log_nr.txt

echo 'run si...'
python cl_si.py --epoch 6 --c 11 --elambda 0.0007 >> log_si.txt

echo 'run gem with buffer size: 128...'
python cl_gem.py --epoch 6 --bsize 128 >> log_gem_128.txt

echo 'run gem with buffer size: 512...'
python cl_gem.py --epoch 6 --bsize 512 >> log_gem_512.txt

echo 'run gem with buffer size: 1024...'
python cl_gem.py --epoch 6 --bsize 1024 >> log_gem_1024.txt

echo 'run tc-pnn...'
python cl_tcpnn.py --epoch 6 >> log_tcpnn.txt
