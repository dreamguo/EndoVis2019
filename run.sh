#my.sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 main.py --feature_type flow_oversample_4 --use_tf_log 1 --train 1 --train 2 --train 3 --train 4 --train 5 --train 6 --train 7 --train 8 --train 9 --train 10 --train 11 --train 12 --train 13 --train 14 --train 15 --train 16 --train 17 --train 18 --test 19 --test 20 --test 21 --test 22 --test 23 --test 24

#CUDA_VISIBLE_DEVICES=1 python3 main.py --feature_type flow_oversample_4_norm --use_tf_log 1 --train 1 --train 2 --train 3 --train 4 --train 5 --train 6 --train 7 --train 8 --train 9 --test 10 --test 11 --test 12

#CUDA_VISIBLE_DEVICES=1 python3 main.py --feature_type flow_oversample_4 --use_tf_log 1 --train 1 --test 10 --test 11

