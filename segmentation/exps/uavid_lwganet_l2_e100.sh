CUDA_VISIBLE_DEVICES=2 python ../train_supervision.py -c ../config/uavid/unetformer_decouplenet_d2_e100.py
CUDA_VISIBLE_DEVICES=2 python ../inference_uavid.py -c ../config/uavid/unetformer_decouplenet_d2_e100.py -o ../fig_results/uavid/unetformer_decouplenet_d2_e100 -t 'lr' -ph 1024 -pw 1024 -b 2 -d "uavid"
