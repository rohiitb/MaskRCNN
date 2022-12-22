import torch
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import *
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset

        self.images = h5py.File(path["images"],'r')['data']
        self.masks = h5py.File(path["masks"],'r')['data']
        self.bboxes = np.load(path["bboxes"], allow_pickle=True)
        self.labels = np.load(path["labels"], allow_pickle=True)
        
        self.corresponding_masks = []
        # Aligning masks with labels
        count = 0
        for i, label in enumerate(self.labels):
            n = label.shape[0] 
            temp = []
            for j in range(n):
                temp.append(self.masks[count])
                count += 1
            self.corresponding_masks.append(temp)

        # Applying rescaling and mean, std, padding
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((800, 1066)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.resize = torchvision.transforms.Resize((800, 1066))
        self.pad = torch.nn.ZeroPad2d((11,11,0,0))

        #############################################


    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index

        one_label = self.labels[index]
        one_image = self.images[index]
        one_bbox = self.bboxes[index]
        one_mask = self.corresponding_masks[index]

        one_label = torch.tensor(one_label)
        one_image, one_mask, one_bbox = self.pre_process_batch(one_image, one_mask, one_bbox)
        ################################

        assert one_image.shape == (3,800,1088)
        assert one_bbox.shape[0] == one_mask.shape[0]

        return one_image, one_label, one_mask, one_bbox, index
        

    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, box):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes

        img_normalized = img / 255.                     # Normalized between 0 & 1
        img_normalized = torch.tensor(img_normalized, dtype=torch.float)   # Converted to tensor
        img_scaled = self.transform(img_normalized)    # Rescaled to (800, 1066) and adjusted for given mean and std

        img_final = self.pad(img_scaled)              # Padded with zeros to get the shape as (800, 1088)

        msk_final = torch.zeros((len(mask), 800, 1088))           #Initializing mask tensor

        for i, msk in enumerate(mask):
            msk = msk/1.                                  # Converting it to uint8
            msk = torch.tensor(msk, dtype=torch.float).view(1,300,400)         # Converting it to tensor
            msk_scaled = self.pad(self.resize(msk).view(800,1066))            # Padding and resizing
            msk_scaled[msk_scaled < 0.5] = 0
            msk_scaled[msk_scaled > 0.5] = 1
            msk_final[i] = msk_scaled

        box = torch.tensor(box, dtype=torch.float)
        box_final = torch.zeros_like(box)
        box_final[:,1] = box[:,1] * 800/300                  # Scaling x
        box_final[:,3] = box[:,3] * 800/300                  # Scaling x
        box_final[:, 0] = box[:,0] * (1066/400) + 11                # Scaling y
        box_final[:, 2] = box[:,2] * (1066/400) + 11                # Scaling y
        
        return img_final, msk_final, box_final
        ######################################
    
    def __len__(self):
        return len(self.labels)


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        out_batch = {}
        out_batch["images"] = []
        out_batch["labels"] = []
        out_batch["masks"] = []
        out_batch["bboxes"] = []
        out_batch["index"] = []

        for i in batch:
            out_batch["images"].append(i[0])
            out_batch["labels"].append(i[1])
            out_batch["masks"].append(i[2])
            out_batch["bboxes"].append(i[3])
            out_batch["index"].append(i[4])

        out_batch["images"] = torch.stack(out_batch["images"], dim=0)
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
    # file path and make a list
    path = {}
    path["images"] = "./data/hw3_mycocodata_img_comp_zlib.h5" 
    path["labels"] = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    path["bboxes"] = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    path["masks"] = "./data/hw3_mycocodata_mask_comp_zlib.h5"
    # load the data into data.Dataset
    dataset = BuildDataset(path)

  
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    for i,batch in enumerate(train_loader,0):
        images=batch['images']
        indexes=batch['index']
        boxes=batch['bboxes']

        gt1,ground_coord1=rpn_net.create_batch_truth1(boxes,indexes,images.shape[-2:])
        gt2,ground_coord2=rpn_net.create_batch_truth2(boxes,indexes,images.shape[-2:])

        print("coord : ", torch.where(ground_coord1 != ground_coord2))
        print("clas : ", torch.where(gt1 != gt2))


        # Flatten the ground truth and the anchors
        flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord1,gt1,rpn_net.get_anchors())

        
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)
        fig,ax=plt.subplots(1,1)

        ax.imshow(images.squeeze(0).permute(1,2,0))
        
        find_cor=(flatten_gt==1).nonzero()
        find_neg=(flatten_gt==-1).nonzero()
             
        for elem in find_cor:
            coord=decoded_coord[elem,:].view(-1)
            # print("coord : ", coord)
            anchor=flatten_anchors[elem,:].view(-1)

            col='r'
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)

        plt.show()
 
        if(i>20):
            break
        

 