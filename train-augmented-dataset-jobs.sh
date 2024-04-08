python train.py -n mixed-resolutions -p augmented ShipRSImageNet_V1_Augmented
python train.py -e 300 -n mixed-resolutions -p augmented ShipRSImageNet_V1_Augmented
python train.py -n mixed-resolutions -p class-weighting-augmented -c ShipRSImageNet_V1_Augmented

python train.py -n matched-resolutions -p augmented ShipRSImageNet_V1_Augmented_MatchedRes
python train.py -e 300 -n matched-resolutions -p augmented ShipRSImageNet_V1_Augmented_MatchedRes
python train.py -n matched-resolutions -p class-weighting-augmented -c ShipRSImageNet_V1_Augmented_MatchedRes

python train.py -n matched-resolutions-moreinstances -p augmented ShipRSImageNet_V1_Augmented_MatchedRes_MoreInstances
python train.py -e 300 -n matched-resolutions-moreinstances -p augmented ShipRSImageNet_V1_Augmented_MatchedRes_MoreInstances
python train.py -n matched-resolutions-moreinstances -p class-weighting-augmented -c ShipRSImageNet_V1_Augmented_MatchedRes_MoreInstances
