import os
import augly.image as imaugs

accepted_datasets = ["cifar10_train", "cifar100_train", "svhn_train",
                     "mnist_train", "fashion_mnist_train", "gtsrb_train",
                     "cifar10_test", "cifar100_test", "svhn_test", "fractal_test",
                     "mnist_test", "fashion_mnist_test", "emnist_test", "asl_mnist_test", "simpsons_mnist_test",
                     "tiny_imagenet_test", "lsun_test", "imagenet_test"]

mean_transform = {
    "resnet": [0.4914, 0.4822, 0.4465],
    "densenet": [125.3 / 255, 123.0 / 255, 113.9 / 255],
    "cnn": []
}
std_transform = {
    "resnet": [0.2023, 0.1994, 0.2010],
    "densenet": [63.0 / 255, 62.1 / 255.0, 66.7 / 255.0],
    "cnn": []
}

additional_transforms = {
    "brightness": imaugs.Brightness(factor=5),
    "blur": imaugs.Blur(radius=4),
    "pixelization": imaugs.Pixelization(ratio=0.1),
    "shuffle_pixels": imaugs.ShufflePixels(factor=0.3),
    "rotate": imaugs.Rotate(degrees=25),
    "contrast": imaugs.Contrast(factor=9.0),
    "opacity": imaugs.Opacity(level=0.2),
    "saturation": imaugs.Saturation(factor=17.0)
}

accepted_attacks = ["fgsm", "deepfool", "pgd"]

datasets_path = "./Data"
if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

path_asl_mnist = datasets_path + "/ASL_MNIST/"
path_simpsons_mnist = datasets_path + "/simpsonsMNIST/"
path_tinyImagenet = datasets_path + "/Imagenet_resize/"
path_lsun = datasets_path + "/LSUN_resize2/"
path_fractal = datasets_path + "/fractal256/"

downloadUrl_tinyImagenet = "https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz"
downloadUrl_lsun = "https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz"
