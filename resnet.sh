#my.sh
#!/bin/bash

#CUDA_VISIBLE_DEVICES=2 python3 extract_resnet_new.py --video_dir ../New_input/video --output_dir ../resnet_output/oversample_4/video/ --mode_type oversample --batch_size 2048 --frame_rate 4
CUDA_VISIBLE_DEVICES=2 python3 extract_resnet_new.py --video_dir ../New_input/video --output_dir ../resnet_output/center_4/video/ --mode_type center --batch_size 2048 --frame_rate 4

