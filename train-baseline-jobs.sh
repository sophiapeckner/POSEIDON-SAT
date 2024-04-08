python train.py -p baselines ShipRSImageNet_V1
python train.py -e 300 -p baselines ShipRSImageNet_V1
python train.py -p baselines -n train-class-weighting -c ShipRSImageNet_V1
python train.py -e 300 -p baselines -n train-class-weighting -c ShipRSImageNet_V1