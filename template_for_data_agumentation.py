import glob
import os
import torchvision

# data agumentation
def data_agument(origin_data_path, agu_data_path):
    imgs = glob.glob(os.path.join(origin_data_path, '*.jpg'))
    i = 0
    for img in imgs:
        # get data
        origin_image_path = img
        origin_lable_path = origin_image_path.replace('image', 'label').replace('jpg', 'png')
        agu_image_path = agu_data_path
        agu_label_path = agu_data_path.replace('image', 'label')
        image = torchvision.io.read_image(origin_image_path, mode=torchvision.io.ImageReadMode.RGB)
        label = torchvision.io.read_image(origin_lable_path, mode=torchvision.io.ImageReadMode.GRAY)

        # original data
        i = i + 1
        torchvision.io.write_png(image, agu_data_path + '{}.png'.format(i))
        torchvision.io.write_png(label, agu_label_path + '{}.png'.format(i))
        
        # vertical flip
        i = i + 1
        image_v = torchvision.transforms.RandomVerticalFlip(p=1)(image)
        label_v = torchvision.transforms.RandomVerticalFlip(p=1)(label)
        torchvision.io.write_png(image_v, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_v, agu_label_path + '{}.png'.format(i))

        # horizontal filp
        i = i + 1
        image_h = torchvision.transforms.RandomHorizontalFlip(p=1)(image)
        label_h = torchvision.transforms.RandomHorizontalFlip(p=1)(label)
        torchvision.io.write_png(image_h, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_h, agu_label_path + '{}.png'.format(i))

        # vertical filp and horizontal filp
        i = i + 1
        image_vh = torchvision.transforms.RandomHorizontalFlip(p=1)(image)
        label_vh = torchvision.transforms.RandomHorizontalFlip(p=1)(label)
        torchvision.io.write_png(image_vh, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_vh, agu_label_path + '{}.png'.format(i))

        # brightness increase
        i = i + 1
        image_bi = torchvision.transforms.ColorJitter(brightness=1.5)(image)
        label_bi = label
        torchvision.io.write_png(image_bi, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_bi, agu_label_path + '{}.png'.format(i))

        # brightness decline
        i = i + 1
        image_bd = torchvision.transforms.ColorJitter(brightness=0.5)(image)
        label_bd = label
        torchvision.io.write_png(image_bd, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_bd, agu_label_path + '{}.png'.format(i))

        # contrast adjustment increase
        i = i + 1
        image_ci = torchvision.transforms.ColorJitter(contrast=1.5)(image)
        label_ci = label
        torchvision.io.write_png(image_ci, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_ci, agu_label_path + '{}.png'.format(i))

        # contrast adjustment decline
        i = i + 1
        image_cd = torchvision.transforms.ColorJitter(contrast=0.5)(image)
        label_cd = label
        torchvision.io.write_png(image_cd, agu_image_path + '{}.png'.format(i))
        torchvision.io.write_png(label_cd, agu_label_path + '{}.png'.format(i))

if __name__ == "__main__":
    origin_data_path = '/root/autodl-tmp/u-net-pytroch-v2_5/dataset/train/origin_data/image/'
    agu_data_path = '/root/autodl-tmp/u-net-pytroch-v2_5/dataset/train/agu_data/image/'
    data_agument(origin_data_path=origin_data_path, agu_data_path=agu_data_path)
