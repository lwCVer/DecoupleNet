CUDA_VISIBLE_DEVICES=1 python train_supervision.py -c config/loveda/unetformer_decouplenet_d2_e50.py
CUDA_VISIBLE_DEVICES=1 python loveda_test.py -c config/loveda/unetformer_decouplenet_d2_e50.py -o fig_results/loveda/unetformer_decouplenet_d2_e50 -t 'd4'

