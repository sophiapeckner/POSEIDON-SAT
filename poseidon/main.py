from generation.coco_instance_generator import COCOInstanceGenerator
from extraction.coco_instance_extractor import COCOInstanceExtractor
from utils.normalization import normalize
from utils.coco2yolo import COCO2YOLO

if __name__ == '__main__':
    
    with open('../selected_for_augmentation.txt', 'r') as f:
        images_to_process = f.read().splitlines()
    
    print(f'Augmenting {len(images_to_process)} images...')
    
    print('Normalizing images...')
    normalize("../ShipRSImageNet_V1", "../ShipRSImageNet_V1_Normalized", images_to_process, 'train_level_2')

    #extractor = COCOInstanceExtractor()
    #extractor.dataset_stats()
    #extractor.extract()

    #generator = COCOInstanceGenerator()
    #generator.balance('/Users/pabloruizponce/Vainas/POSEIDON/poseidon/outputs')
    
    #conversor = COCO2YOLO()
    #conversor.convert("/Users/pabloruizponce/Vainas/SDSYOLO", "SDSYOLO")

    #conversor = COCO2YOLO(augmented=True)
    #conversor.convert("/Users/pabloruizponce/Vainas/SDSYOLOAugmented", "SDSYOLOAugmented")

