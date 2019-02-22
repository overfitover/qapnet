import os

import sys
sys.path.append("/home/yxk/project/aa_demo_graduate/qapNet")
sys.path.append("/home/yxk/project/aa_demo_graduate/qapNet/wavedata")


import numpy as np
import avod
from avod.builders.dataset_builder import DatasetBuilder



def do_preprocessing(dataset, indices):

    mini_batch_utils = dataset.kitti_utils.mini_batch_utils

    print("Generating mini batches in {}".format(
        mini_batch_utils.mini_batch_dir))

    # Generate all mini-batches, this can take a long time
    mini_batch_utils.preprocess_rpn_mini_batches(indices)

    print("Mini batches generated")

def main(dataset=None):
    """Generates anchors info which is used for mini batch sampling.

    Processing on 'Cars' can be split into multiple processes, see the Options
    section for configuration.

    Args:
        dataset: KittiDataset (optional)
            If dataset is provided, only generate info for that dataset.
            If no dataset provided, generates info for all 3 classes.
    """

    if dataset is not None:
        do_preprocessing(dataset, None)
        return

    car_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/rpn_cars.config'
    ped_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/rpn_pedestrians.config'
    cyc_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/rpn_cyclists.config'
    ppl_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/rpn_people.config'

    ##############################
    # Options
    ##############################
    # Serial vs parallel processing
    in_parallel = False

    process_car = True   # Cars
    process_ped = False  # Pedestrians
    process_cyc = False  # Cyclists
    process_ppl = False   # People (Pedestrians + Cyclists)


    ##############################
    # Dataset setup
    ##############################
    if process_car:
        car_dataset = DatasetBuilder.load_dataset_from_config(
            car_dataset_config_path)
    if process_ped:
        ped_dataset = DatasetBuilder.load_dataset_from_config(
            ped_dataset_config_path)
    if process_cyc:
        cyc_dataset = DatasetBuilder.load_dataset_from_config(
            cyc_dataset_config_path)
    if process_ppl:
        ppl_dataset = DatasetBuilder.load_dataset_from_config(
            ppl_dataset_config_path)

    ##############################
    # Serial Processing
    ##############################
    if not in_parallel:
        if process_car:
            do_preprocessing(car_dataset, [1])
        if process_ped:
            do_preprocessing(ped_dataset, [1])
        if process_cyc:
            do_preprocessing(cyc_dataset, [1])
        if process_ppl:
            do_preprocessing(ppl_dataset, [1])

        print('All Done (Serial)')


if __name__ == '__main__':
    main()

