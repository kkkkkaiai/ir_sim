import glob
import imageio
import os

#def save_animate(self, ani_name='animated', suffix='.gif', keep_len=30, rm_fig_path=True, **kwargs):
name = 'result3'
suffix = '.gif'
animation_dir = './'+name
images = list(glob.glob(os.path.join(animation_dir, '*.png')))
images.sort()
image_list = []
for i, file_name in enumerate(images):
    if i == 0: continue

    image_list.append(imageio.imread(str(file_name)))
    # if i == len(images) - 1:
    #     for j in range(keep_len):
    #         image_list.append(imageio.imread(str(file_name)))

imageio.mimsave('./' + name + suffix, image_list)
# print('Create animation successfully, the animation file is saved in the path ' + str(self.ani_path))
