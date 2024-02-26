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

    print('Compiling source instances for generator...')
    extractor = InstanceExtractor(original_dataset)
    extractor.extract_instances(CLASS_TO_AUGMENT, 'fishing_vessel_instances')
    print()

    # See notebook for selection of these args
    target_avg_of_instances_per_image = 3
    total_instances_to_add = len(images_to_process) * target_avg_of_instances_per_image
    min_instances_per_image = 1
    max_instances_per_image = 5

    print(f'Augmenting images...')
    generator = InstanceGenerator('Fishing Vessel', 'fishing_vessel_instances')
    generator.augment(original_dataset, "ShipRSImageNet_V1_Augmented", images_to_process,
                      total_instances_to_add,
                      min_instances_per_image,
                      max_instances_per_image,
                     ) 
    print()
