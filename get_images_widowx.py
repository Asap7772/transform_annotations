import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc

dataset_path = '/media/ashvin/data2/bridge_data_numpy_shifted_split_uint8_4-15'
kitchens_to_sample_from = ['toykitchen1']
widowx_images_path='/home/anikaitsingh/transform_annotations/widowx_images'

train_tasks = val_tasks = [
    'put_broccoli_in_pot_cardboardfence',
    'put_carrot_on_plate_cardboardfence',
    'put_broccoli_in_pot_or_pan',
    'put_broccoli_in_bowl',
    'put_carrot_on_plate',
    'put_sushi_on_plate',
    'put_corn_into_bowl',
    'put_sweet_potato_in_pan_which_is_on_stove',
    'put_sweet_potato_in_pan_which_is_on_stove_distractors',
    'put_sweet_potato_in_pot_which_is_in_sink_distractors',

    'take_broccoli_out_of_pan_cardboardfence',
    'take_carrot_off_plate_cardboardfence',
    'take_broccoli_out_of_pan',
    'take_can_out_of_pan',
    'take_carrot_off_plate',
    'take_lid_off_pot_or_pan',
]

paths_to_glob=[os.path.join(dataset_path, x) for x in kitchens_to_sample_from]
paths_to_glob=[glob.glob(x+'/*') for x in paths_to_glob if os.path.exists(x)]

all_paths = []
for x in paths_to_glob:
    all_paths.extend(x)

all_paths=[x for x in all_paths if x.split('/')[-1] in train_tasks]

all_paths = [os.path.join(x, 'train', 'out.npy') for x in all_paths]
for x in all_paths:
    assert os.path.exists(x), f'{x} does not exist'

def sample_images(path, num=2):
    # choose first image from separate trajectories
    data = np.load(path, allow_pickle=True)

    which_traj = np.random.choice(np.arange(len(data)), size=(num,), replace=False)
    
    images = []
    for i in range(num):
        try:
            images.append(data[which_traj][i]['observations'][0]['images0'])
        except:
            images.append(data[which_traj][i]['observations'][0]['images'])
    return images

all_images = []
for x in all_paths:
    print(f'Sampling from {x}')
    all_images.extend(sample_images(x))
    gc.collect()

os.makedirs(widowx_images_path, exist_ok=True)
for i in range(len(all_images)):
    path = os.path.join(widowx_images_path,'image_' + str(i) + '.png') 
    im = Image.fromarray(all_images[i])
    im.save(path)
