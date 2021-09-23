#!/bin/bash

# run without neptune logging.
echo 'run the fine-tune...'
python cl_keyword.py --epoch 10

echo 'run the naive rehearsal...'
python cl_rehearsal.py --epoch 10 --ratio 1

echo 'run ewc...'
python cl_ewc.py --epoch 10 --elambda 9

echo 'run si...'
python cl_si.py --epoch 10 --c 11 --elambda 0.0007

echo 'run gem-128...'
python cl_gem.py --epoch 10 --bsize 128

echo 'run gem-640...'
python cl_gem.py --epoch 10 --bsize 640

echo 'run tc-pnn...'
python cl_tcpnn.py --epoch 10
