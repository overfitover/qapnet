import os


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    avod_root_dir = root_dir()
    return os.path.split(avod_root_dir)[0]


if __name__ == '__main__':
    print(root_dir())   # /home/yxk/project/aa_demo_graduate/qapNet/avod
    print(top_dir())    # /home/yxk/project/aa_demo_graduate/qapNet
