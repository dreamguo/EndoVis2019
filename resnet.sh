#my.sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=2 time python3 extract_ResNet.py --video_dir ../New_input/video --output_dir ../resnet_output/video/ --mode_type oversample

