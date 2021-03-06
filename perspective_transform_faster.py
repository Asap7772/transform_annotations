import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

images_path = './widowx_images'
all_images = sorted([os.path.join(os.path.abspath(images_path), x) for x in os.listdir(images_path) if x.endswith('.png')])

num_aug = 5
range_aug = 3

# if True:
    # all_images = all_images[:2]

all_matrices = []

pts_dict = {}
for x in all_images:
    print('loading image', x)
    img1 = plt.imread(x)
    # show image1
    plt.figure(figsize=(10,10))
    plt.imshow(img1)
    pts = plt.ginput(4)
    pts = np.array(pts,dtype=np.float32)
    print('points', pts)
    plt.close()
    pts_dict[x] = pts


for x in all_images:
    for y in all_images:
        if x != y: # we don't want to compare the same image to itself but order matters
            print(f'shift from {x} to {y}')
            img1 = plt.imread(x)
            img2 = plt.imread(y)
            
            pts1 = pts_dict[x]
            pts2 = pts_dict[y]
            
            matrices = [cv2.getPerspectiveTransform(pts1, pts2)]
            
            for k in range(num_aug):
                mod_pts1 = pts1 + np.random.randint(-range_aug, range_aug, size=pts1.shape).astype(np.float32)
                mod_pts2 = pts2 + np.random.randint(-range_aug, range_aug, size=pts1.shape).astype(np.float32)
                matrices.append(cv2.getPerspectiveTransform(mod_pts1, mod_pts2))                   
            all_matrices.extend(matrices)
            
            image1_cv = cv2.imread(x)
            result = cv2.warpPerspective(image1_cv, matrices[0], (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(1,3,figsize=(25,10))
            ax[0].imshow(result)
            ax[1].imshow(img2)
            a1 = ax[2].scatter(pts1[:,0], pts1[:,1])
            a2 = ax[2].scatter(pts2[:,0], pts2[:,1])
            ax[2].legend((a1, a2), ('original', 'transformed'))
            plt.show()
            
# save all matrices
out_path = './matrices/'
os.makedirs(out_path, exist_ok=True)
file_name = 'matrices_widowx.npy'
full_path = os.path.join(os.path.abspath(out_path), file_name)
print(f'Saving matrices to {full_path}')
np.save(full_path, all_matrices)

# load all matrices
arr = np.load(full_path, allow_pickle=True)
import ipdb; ipdb.set_trace()
