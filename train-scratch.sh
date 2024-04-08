python train.py -e 300 -p baselines -n scratch --no-pretrained ShipRSImageNet_V1
python train.py -e 300 -p baselines -n scratch-class-weighting -c --no-pretrained ShipRSImageNet_V1

python train.py -e 300 -n scratch-mixed-resolutions -p augmented --no-pretrained ShipRSImageNet_V1_Augmented
python train.py -e 300 -n scratch-mixed-resolutions -p class-weighting-augmented -c --no-pretrained ShipRSImageNet_V1_Augmented

python train.py -e 300 -n scratch-matched-resolutions -p augmented --no-pretrained ShipRSImageNet_V1_Augmented_MatchedRes
python train.py -e 300 -n scratch-matched-resolutions -p class-weighting-augmented -c --no-pretrained ShipRSImageNet_V1_Augmented_MatchedRes

python train.py -e 300 -n scratch-matched-resolutions-moreinstances -p augmented --no-pretrained ShipRSImageNet_V1_Augmented_MatchedRes_MoreInstances
python train.py -e 300 -n scratch-matched-resolutions-moreinstances -p class-weighting-augmented -c --no-pretrained ShipRSImageNet_V1_Augmented_MatchedRes_MoreInstances
