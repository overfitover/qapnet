import avod
import sys
sys.path.append('{}'.format(avod.top_dir()))
sys.path.append('{}'.format(avod.top_dir())+'/wavedata')
sys.path.append('{}'.format(avod.root_dir()))
from avod.builders.dataset_builder import DatasetBuilder

def main(dataset=None):
    if not dataset:
        dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_TRAIN)

    label_cluster_utils = dataset.kitti_utils.label_cluster_utils

    print("Generating clusters in {}/{}".format(
        label_cluster_utils.data_dir, dataset.data_split))  # /home/yxk/project/aa_demo_graduate/qapNet/avod/data/label_clusters/train
    clusters, std_devs = dataset.get_cluster_info()

    print("Clusters generated")
    print("classes: {}".format(dataset.classes))            # ['car']
    print("num_clusters: {}".format(dataset.num_clusters))  # [2]
    print("all_clusters:\n {}".format(clusters))            # [[array([3.513, 1.581, 1.511]), array([4.234, 1.653, 1.546])]]
    print("all_std_devs:\n {}".format(std_devs))            # [[array([0.254, 0.097, 0.112]), array([0.25 , 0.102, 0.156])]]


if __name__ == '__main__':
    main()
