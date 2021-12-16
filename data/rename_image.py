import os
import shutil
from os.path import join

stages = ['train', 'val']

for target_root in stages:
	image_names = os.listdir(target_root)
	print(image_names)
	img_des = './new_{}/images'.format(target_root)
	ann_des = './new_{}/masks'.format(target_root)

	if not os.path.exists(img_des):
		os.makedirs(img_des)
	if not os.path.exists(ann_des):
		os.makedirs(ann_des)
		
	for img_name in image_names:
		# move source images
		img_dir = join(target_root, img_name, 'images')
		img_source = join(img_dir, os.listdir(img_dir)[0])
		shutil.move(img_source, img_des)

		# rename and move source images
		ann_dir = join(target_root, img_name, 'masks')
		mask_imgs = os.listdir(ann_dir)  # mask images
		for i,mask_img in enumerate(mask_imgs):
			if mask_img[:4] == 'mask':
				src = join(ann_dir, mask_img)
				des = ('{}/{}_{}.png'.format(ann_des, img_name, i))
				os.rename(src, des)