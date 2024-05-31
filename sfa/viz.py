import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def overlay_lidar_on_img(args):
    name = args.img_name
    dataset_dir = args.dataset_dir
    mode = args.mode
    save_dir = args.save_dir

    img_path = os.path.join(dataset_dir, mode, 'image_2', f'{name}.png')
    binary_path = os.path.join(dataset_dir, mode, 'velodyne', f'{name}.bin')
    calib_path = os.path.join(dataset_dir, mode, 'calib', f'{name}.txt')
    with open(calib_path,'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    # read raw data from binary
    scan = np.fromfile(binary_path, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    # TODO: use fov filter? 
    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    png = mpimg.imread(img_path)
    IMG_H,IMG_W,_ = png.shape

    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    # plt.imshow(png)

    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)

    # generate color map from depth
    u,v,z = cam
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    plt.title(name)
    plt.savefig(os.path.join(save_dir, 'overlay_lidar_on_img.png'),bbox_inches='tight')
    # plt.show()


def overlay_bev_on_img(name):
    # sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
    # name = '%06d'%sn # 6 digit zeropadding
    img_path = f'../dataset/kitti/testing/image_2/{name}.png'
    bev_map_path = f'../results/validation/bev/{name}.npy'
    binary_path = f'../dataset/kitti/testing/velodyne/{name}.bin'
    with open(f'../dataset/kitti/testing/calib/{name}.txt','r') as f:
        calib = f.readlines()

    img = mpimg.imread(img_path)
    IMG_H, IMG_W, _ = img.shape
    bev_map = np.load(bev_map_path)

    # P2 (3 x 4) for left eye
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    # # TODO: use fov filter?
    # velo = np.insert(points,3,1,axis=1).T
    # velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    # cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
    # cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    # # Get u, v, z
    # cam[:2] /= cam[2,:]
    
    # # Filter points outside the image
    # u,v,z = cam
    # u_out = np.logical_or(u<0, u>IMG_W)
    # v_out = np.logical_or(v<0, v>IMG_H)
    # outlier = np.logical_or(u_out, v_out)
    # cam = np.delete(cam,np.where(outlier),axis=1)
    # u,v,z = cam

    # Create the RGB image from BEV map
    bev_rgb_image = create_bev_rgb_image(bev_map)
    
    bev_resized = cv2.resize(bev_rgb_image, (IMG_W, IMG_H))

    # Convert the image to float32 for blending
    image_float = img.astype(np.float32) / 255.0
    bev_float = bev_resized.astype(np.float32)

    # Blend the images
    overlay = cv2.addWeighted(image_float, 1.0, bev_float, 0.5, 0.0)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    # Display the result
    plt.imshow(cv2.cvtColor(overlay_uint8, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis
    plt.title('Overlay of BEV Map on 2D Image')
    plt.savefig(f'{name}.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    
    overlay_bev_on_img('003712')