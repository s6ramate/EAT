import os
import random
import glob
import numpy as np
import torch
import yaml
import pickle
from util.data_util import data_prepare

from PIL import Image
from torchvision.transforms import ToTensor,Resize, Compose, Normalize, ToPILImage


#Elastic distortion
def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0)//gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag


class SemanticKITTI(torch.utils.data.Dataset):
    def __init__(self, 
        data_path, 
        target = "pointcloud",
        use_pseudo_voxels=False,
        voxel_size=[0.1, 0.1, 0.1], 
        split='train', 
        return_ref=True, 
        label_mapping="util/semantic-kitti.yaml", 
        rotate_aug=True, 
        flip_aug=True, 
        scale_aug=True, 
        scale_params=[0.95, 1.05], 
        transform_aug=True, 
        trans_std=[0.1, 0.1, 0.1],
        elastic_aug=False, 
        elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
        ignore_label=255, 
        voxel_max=None, 
        xyz_norm=False, 
        pc_range=None, 
        use_tta=None,
        vote_num=4,
    ):
        super().__init__()
        self.num_classes = 19
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.target = target
        self.use_pseudo_voxels = use_pseudo_voxels
        self.learning_map = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml['learning_map_inv']
        self.remap_lut = self.get_remap_lut()
        self.inv_remap_lut = self.get_inv_remap_lut()
        self.return_ref = return_ref
        self.split = split
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.scale_params = scale_params
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.elastic_aug = elastic_aug
        self.elastic_gran, self.elastic_mag = elastic_params[0], elastic_params[1]
        self.use_tta = use_tta
        self.vote_num = vote_num
        
        self.transform = Compose([
            ToTensor(),
            Resize((11*32, 38*32)),
            Normalize(mean = [0.3502, 0.3689, 0.3670], std = [0.3017, 0.3100, 0.3182]),
        ])


        if split == 'train':
            splits = semkittiyaml['split']['train']
        elif split == 'val':
            splits = semkittiyaml['split']['valid']
        elif split == 'test':
            splits = semkittiyaml['split']['test']
        elif split == 'trainval':
            splits = semkittiyaml['split']['train'] + semkittiyaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')

        self.files = []
        if self.target == "pointcloud":
            for i_folder in splits:
                self.files += sorted(glob.glob(os.path.join(data_path, "sequences", str(i_folder).zfill(2), 'velodyne', "*.bin")))
        elif self.target == "voxel":
            for i_folder in splits:
                self.files += sorted(glob.glob(os.path.join(data_path, "sequences", str(i_folder).zfill(2), 'voxels', "*[05].bin")))
            self.files = list(map(lambda x: x.replace("voxels", "velodyne"),self.files))
        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.nusc_infos)
        return len(self.files)

    def __getitem__(self, index):
        if self.use_tta:
            samples = []
            for i in range(self.vote_num):
                sample = tuple(self.get_single_sample(index, vote_idx=i))
                samples.append(sample)
            return tuple(samples)
        return self.get_single_sample(index)

    def get_single_sample(self, index, vote_idx=0):

        file_path = self.files[index]

        raw_data = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        
        if self.split != 'test' or self.target != "voxel":        
            annotated_data = np.fromfile(file_path.replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        points = raw_data[:, :4]

        if self.split != 'test':
            annotated_data[annotated_data == 0] = self.ignore_label + 1
            annotated_data = annotated_data - 1
            labels_in = annotated_data.astype(np.uint8).reshape(-1)
        else:
            labels_in = np.zeros(points.shape[0]).astype(np.uint8)


        feats = points
        xyz = points[:, :3]
        
        if self.pc_range is not None:            
            
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1])

        scales = [2**i for i in range(5)]
        scale_names = list(map(lambda x : f"1_{x}", scales))
        scales_and_names = list(zip(scales, scale_names))
        
        image = self.get_image(file_path)
        if self.split == 'train':
            coords, xyz, feats, labels = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            if(self.target == "voxel"):
                labels = {scale_name: self.get_label_at_scale(file_path,scale, occ= "occ" in scale_name, filtering_label=labels_in) for scale, scale_name in scales_and_names}
            return coords, xyz, feats, labels, self.files[index], image
        else:
            coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            if(self.target == "voxel"):
                labels = {scale_name: self.get_label_at_scale(file_path,scale, occ= "occ" in scale_name, filtering_label=labels_in) for scale, scale_name in scales_and_names}
                
            if self.split == 'val':
                return coords, xyz, feats, labels, inds_reconstruct, self.files[index], image
            elif self.split == 'test':
                return coords, xyz, feats, labels, inds_reconstruct, self.files[index], image
  
  
    def get_image(self, filepath):
        head, filename = os.path.split(filepath)
        import re
        
        filenum = int(re.sub("\D","",filename))
        if filenum > 0:
            images = []
            for i in range(filenum - 4, filenum+1):
                filepath = os.path.join(head, f"{filenum:06d}"+filename[-4:])
                filepath1 = filepath.replace("velodyne", "image_2")
                filepath1 = filepath1.replace("bin", "png")
                filepath2 = filepath.replace("velodyne", "image_3")
                filepath2 = filepath2.replace("bin", "png")
                
                img1= self.transform(Image.open(filepath1)).unsqueeze(0)
                img2= self.transform(Image.open(filepath2)).unsqueeze(0)
                images.append(img1)
                images.append(img2)
            return torch.cat(images, dim =1)

        else :
            filepath1 = filepath.replace("velodyne", "image_2")
            filepath1 = filepath1.replace("bin", "png")

            filepath2 = filepath.replace("velodyne", "image_3")
            filepath2 = filepath2.replace("bin", "png")
            
            img1= self.transform(Image.open(filepath1)).unsqueeze(0)
            img2= self.transform(Image.open(filepath2)).unsqueeze(0)
            return torch.cat((img1,img2,), dim =1).repeat((1,5,1,1))

    def get_remap_lut(self):
        """
        remap_lut to remap classes of semantic kitti for training...
        :return:
        """

        # make lookup table for mapping
        maxkey = max(self.learning_map.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.learning_map.keys())] = list(
            self.learning_map.values()
        )

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    def get_inv_remap_lut(self):
        """
        remap_lut to remap classes of semantic kitti for training...
        :return:
        """

        # make lookup table for mapping
        maxkey = max(self.learning_map_inv.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(self.learning_map_inv.keys())] = list(
            self.learning_map_inv.values()
        )

        return remap_lut


    def get_label(self, filepath):
        filepath = filepath.replace("velodyne", "voxels")
        filepath = filepath.replace("bin", "label")
        
        LABEL = _read_label_SemKITTI(
            filepath
        )
        INVALID = _read_invalid_SemKITTI(
            filepath.replace("label","invalid")
        )
        
        LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(
            np.float32
        )  # Remap 20 classes semanticKITTI SSC
        
        # Setting to unknown all voxels marked on invalid mask...
        LABEL[np.isclose(INVALID, 1)] = 255
        LABEL = np.moveaxis(
            LABEL.reshape(
                [
                    256,
                    256,
                    32,
                ]
            ),
            [0, 1, 2],
            [0, 2, 1],
        )
        # LABEL[LABEL==255] = 0
        return torch.from_numpy(LABEL).unsqueeze(0).long()

    # def get_occ_at_scale(self, filepath, scale=1):
    #     label = self.get_label_at_scale(filepath, scale)
    #     label[(0 < label < 255)] = 1
    #     return label
    
    def label_rectification(grid_ind, voxel_label, instance_label, 
                        dynamic_classes=[1,4,5,6,7,8],
                        voxel_shape=(256,256,32),
                        ignore_class_label=255):
    
        segmentation_label = voxel_label[grid_ind[:,0], grid_ind[:,1], grid_ind[:,2]]
        
        for c in dynamic_classes:
            voxel_pos_class_c = (voxel_label==c).astype(int)
            instance_label_class_c = instance_label[segmentation_label==c].squeeze(1)
            
            if len(instance_label_class_c) == 0:
                pos_to_remove = voxel_pos_class_c
            
            elif len(instance_label_class_c) > 0 and np.sum(voxel_pos_class_c) > 0:
                mask_class_c = np.zeros(voxel_shape, dtype=int)
                point_pos_class_c = grid_ind[segmentation_label==c]
                uniq_instance_label_class_c = np.unique(instance_label_class_c)
                
                for i in uniq_instance_label_class_c:
                    point_pos_instance_i = point_pos_class_c[instance_label_class_c==i]
                    x_max, y_max, z_max = np.amax(point_pos_instance_i, axis=0)
                    x_min, y_min, z_min = np.amin(point_pos_instance_i, axis=0)
                    
                    mask_class_c[x_min:x_max,y_min:y_max,z_min:z_max] = 1
            
                pos_to_remove = (voxel_pos_class_c - mask_class_c) > 0
            
            voxel_label[pos_to_remove] = ignore_class_label
                
        return voxel_label

    def get_label_at_scale(self, filepath, scale=1, occ=False, filtering_label=None):
        
        suffix = "" if scale==1 else f"_1_{scale}"
        filepath = filepath.replace("velodyne", "voxels")
        filepath = filepath.replace("bin", f"label{suffix}")
        
        if self.split == "test":
            LABEL = np.zeros((256*256*32// (scale**3)))
        else:
            LABEL = _read_label_SemKITTI(
                filepath
            )
            
        if self.split == "test":
            INVALID = np.zeros((256*256*32 // (scale**3)))
        else:
            INVALID = _read_invalid_SemKITTI(
                filepath.replace("label",f"invalid")
            )
        

        
        LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(
            np.float32
        )  # Remap 20 classes semanticKITTI SSC
        
        # Setting to unknown all voxels marked on invalid mask...
        LABEL[np.isclose(INVALID, 1)] = 255
        LABEL = np.moveaxis(
            LABEL.reshape(
                [
                    256//scale,
                    256//scale,
                    32//scale,
                ]
            ),
            [0, 1, 2],
            [0, 2, 1],
        )
        # print(np.count_nonzero(LABEL))
        # print(np.count_nonzero(LABEL[LABEL == 255]))
        if occ:
            LABEL[np.logical_and(LABEL!=0,LABEL != 255)] = 1
            
        # LABEL[LABEL==255] = 0
        
        return torch.from_numpy(LABEL).unsqueeze(0).long()

    
def _read_SemKITTI(path, dtype, do_unpack):
    bin = np.fromfile(path, dtype=dtype)  # Flattened array
    if do_unpack:
        bin = unpack(bin)
    return bin

def _read_label_SemKITTI(path):
    label = _read_SemKITTI(path, dtype=np.uint16,
                            do_unpack=False).astype(np.float32)
    return label


def _read_invalid_SemKITTI(path):
    invalid = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
    return invalid

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed
