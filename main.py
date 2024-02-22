from poseidon.generation.instance_generator import InstanceGenerator
from poseidon.extraction.instance_extractor import InstanceExtractor
from poseidon.utils.normalization import normalize
from poseidon.utils.coco2yolo import COCO2YOLO
from shiprsimagenet import ShipRSImageNet


CLASS_TO_AUGMENT = 'Fishing Vessel'


if __name__ == '__main__':
    original_dataset = ShipRSImageNet('ShipRSImageNet_V1')

    with open('selected_for_augmentation.txt', 'r') as f:
        images_to_process = f.read().splitlines()

    # The original normalization method described by POSEIDON is not a good fit for the ShipRSImageNet dataset
    # We'll instead normalize images based on the source and destination spatial resolutions
    #print('Normalizing images...')
    #normalize(original_dataset.root_path, "ShipRSImageNet_V1_Normalized", images_to_process, 'train_level_2')
    #print()

    print('Compiling source images for generator...')
    extractor = InstanceExtractor(original_dataset)
    extractor.extract_instances(CLASS_TO_AUGMENT, 'fishing_vessel_instances')
    print()

    #print(f'Augmenting {len(images_to_process)} images...')
    #generator = COCOInstanceGenerator()
    #generator.balance('/Users/pabloruizponce/Vainas/POSEIDON/poseidon/outputs')
    #print()

    #conversor = COCO2YOLO()
    #conversor.convert("/Users/pabloruizponce/Vainas/SDSYOLO", "SDSYOLO")

    #conversor = COCO2YOLO(augmented=True)
    #conversor.convert("/Users/pabloruizponce/Vainas/SDSYOLOAugmented", "SDSYOLOAugmented")

