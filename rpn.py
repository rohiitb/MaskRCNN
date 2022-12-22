import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from utils import *
from backbone import *
import torchvision
import pytorch_lightning as pl


class RPNHead(pl.LightningModule):
    # _default_cfg = {
    #     # 'device': device,
    #     'anchors_param': dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16),
    #     'batch_size': 4
    # }
    
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, batch_size = 4, num_anchors=3,
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])):
        

        super(RPNHead,self).__init__()

        self.ground_dict={}
        self.num_anchors = num_anchors
        self.anchors_param = anchors_param
        self.batch_size = batch_size
        self.image_size = (800, 1088)
        self.train_loss_epoch = []
        self.train_class_loss_epoch = []
        self.train_reg_loss_epoch = []

        self.val_loss_epoch = []
        self.val_class_loss_epoch = []
        self.val_reg_loss_epoch = []
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'], self.anchors_param['grid_size'],self.anchors_param['stride'])

        self.resnet50_fpn = Resnet50Backbone()

        self.intermediate_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.classifier_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )

        self.regressor_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=12, kernel_size=1, padding='same')
            # nn.Sigmoid()
        )

    def forward(self, X):
        
        # Forward each level of the FPN output through the intermediate layer and the RPN heads
        # Input:
        #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
        # Ouput:
        #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
        #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
       
        
        logits, bbox_regs = MultiApply(self.forward_single, X)

        return logits, bbox_regs


    def forward_single(self, feature):
        # Forward a single level of the FPN output through the intermediate layer and the RPN heads
        # Input:
        #       feature: (bz,256,grid_size[0],grid_size[1])}
        # Ouput:
        #       logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
        #       bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])

        int_out = self.intermediate_layer(feature)

        logit = self.classifier_head(int_out)
        bbox_reg = self.regressor_head(int_out)

        return logit, bbox_reg



    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):
        # This function creates the anchor boxes for one FPN level
        # Input:
        #      aspect_ratio: list:len(number_of_aspect_ratios)
        #      scale: scalar
        #      grid_size: tuple:len(2)
        #      stride: scalar
        # Output:
        #       anchors: (grid_size[0]*grid_size[1]*num_acnhors,4)

        anchors = torch.zeros(self.num_anchors, grid_sizes[0], grid_sizes[1], 4)

        yy, xx = torch.meshgrid(torch.arange(0, grid_sizes[0]*stride, stride, device=device) + (stride/2), torch.arange(0, grid_sizes[1]*stride, stride, device=device) + (stride/2))
        
        # For aspect ratio 1:2
        w0 = scale * np.sqrt(aspect_ratio[0])
        h0 = w0 / aspect_ratio[0]
        anchors[0,:,:,0] = xx
        anchors[0,:,:,1] = yy
        anchors[0,:,:,2] = w0
        anchors[0,:,:,3] = h0

        # For aspect ratio 1:1
        w1 = scale * np.sqrt(aspect_ratio[1])
        h1 = w1 / aspect_ratio[1]
        anchors[1,:,:,0] = xx
        anchors[1,:,:,1] = yy
        anchors[1,:,:,2] = w1
        anchors[1,:,:,3] = h1

        # For aspect ratio 2:1
        w2 = scale * np.sqrt(aspect_ratio[2])
        h2 = w2 / aspect_ratio[2]
        anchors[2,:,:,0] = xx
        anchors[2,:,:,1] = yy
        anchors[2,:,:,2] = w2
        anchors[2,:,:,3] = h2

        ######################################
        # assert anchors.shape == (self.num_anchors, grid_sizes[0], grid_sizes[1], 4)

        return anchors

    def get_anchors(self):
        return self.anchors

    def create_anchors(self, aspect_ratio, scale, grid_size, stride):
        # This function creates the anchor boxes for all FPN level
        # Input:
        #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
        #       scale:        list:len(FPN)
        #       grid_size:    list:len(FPN){tuple:len(2)}
        #       stride:        list:len(FPN)
        # Output:
        #       anchors_list: list:len(FPN){(num_anchors, grid_size[0], grid_size[1], 4)}
        anchors_list = []
        for i in range(5):
          anchors_list.append(self.create_anchors_single(aspect_ratio[i], scale[i], grid_size[i], stride[i]))

        return anchors_list



    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        # This function creates the ground truth for a batch of images
        # Input:
        #      bboxes_list: list:len(bz){(number_of_boxes,4)}
        #      indexes: list:len(bz)
        #      image_shape: list:len(bz){tuple:len(2)}
        # Ouput:
        #      ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
        #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
        image_shape_list = [image_shape for i in range(self.batch_size)]
        anchors_list = [self.get_anchors() for i in range(self.batch_size)]

        grid_size_list = [self.anchors_param["grid_size"] for i in range(self.batch_size)]
        ground_list, ground_coord_list = MultiApply(self.create_ground_truth, bboxes_list, indexes, anchors_list, grid_size_list, image_shape_list)     # Shape len(bz) {len(FPN), (12, grid_size[0], grid_size[1])}
        # ground_list, ground_coord_list = MultiApply(self.create_ground_truth2, bboxes_list, indexes, grid_size_list, anchors_list, image_shape_list)     # Shape len(bz) {len(FPN), (12, grid_size[0], grid_size[1])}


        ground = [torch.cat([each_ground[i].unsqueeze(0) for each_ground in ground_list]) for i in range(5)]
        ground_coord = [torch.cat([each_ground_coord[i].unsqueeze(0) for each_ground_coord in ground_coord_list]) for i in range(5)]

        return ground, ground_coord

    def create_ground_truth2(self, bboxes, index, grid_sizes, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord
        #####################################################
        ground_class = []
        ground_coord = []
        n_fpn = 5
        num_anchors = 3
        img_w, img_h = image_size[1], image_size[0]
        
        # use utils function to process ground truths and anchors
        gt_boxes, anchors_stacked_all = process_for_iou(bboxes, torch.cat(anchors, dim=0), image_size)

        # IOU
        ious = IOU(anchors_stacked_all, gt_boxes)
        
        # remove cross boundary 
        cross_boundary = ((anchors_stacked_all[:, 0] < 0) | (anchors_stacked_all[:, 1] < 0) | (anchors_stacked_all[:, 2] > 1088) | (anchors_stacked_all[:, 3] > 800)).nonzero()[:,0]
        ious[cross_boundary, :] = -1
        
        # find maximum iou
        max_ious = ious.max(dim=0).values
        max_indices = (ious >=  0.999*max_ious).nonzero()
        box_max_indices = max_indices[:,1]
        max_indices = max_indices[:,0] # indices: [a, b, c, ...]
        
        # find the iou > 0.7
        iou_idx_high = (ious > 0.7).nonzero()[:, 0]
        box_iou_idx_high = (ious > 0.7).nonzero()[:, 1]
        
        pos_indices = torch.cat((max_indices, iou_idx_high),0)
        corresponding_boxes = torch.cat((box_max_indices, box_iou_idx_high))

        # find iou < 0.3 and non-positive
        neg_indices = torch.logical_and(ious >= 0, ious < max_ious.clamp(max=0.3)).all(dim=1).nonzero()[:, 0]

        # find what fpn level
        grid_total = [0, 163200, 204000, 214200, 216750, 217413]

        for fpn_idx in range(n_fpn):
            # extract grid size at the given fpn level
            grid_size_fpn = self.grid_size[fpn_idx]
            
            # define tensors for list
            ground_class_one = torch.full(size=(num_anchors, grid_size_fpn[0], grid_size_fpn[1]), fill_value=-1, dtype=torch.float) 
            ground_coord_one = torch.full(size=(4*num_anchors, grid_size_fpn[0], grid_size_fpn[1]), fill_value=-1, dtype=torch.float)
            
            # correct positive index
            pos_indices_fpn_all = pos_indices[torch.logical_and(pos_indices < grid_total[fpn_idx+1],pos_indices >= grid_total[fpn_idx])]
            pos_indices_fpn = pos_indices_fpn_all - grid_total[fpn_idx]
           
            # correct negative index
            neg_indices_fpn_all = neg_indices[torch.logical_and(neg_indices < grid_total[fpn_idx+1],neg_indices >= grid_total[fpn_idx])]
            neg_indices_fpn = neg_indices_fpn_all - grid_total[fpn_idx]
            
            # set class information
            idx = np.unravel_index(pos_indices_fpn, shape=(num_anchors, grid_size_fpn[0], grid_size_fpn[1]))
            ground_class_one[idx[0], idx[1], idx[2]] = 1
            
            idx_neg = np.unravel_index(neg_indices_fpn, shape=(num_anchors, grid_size_fpn[0], grid_size_fpn[1]))
            ground_class_one[idx_neg[0], idx_neg[1], idx_neg[2]] = 0
            
            # set bounding boxes
            corresponding_boxes_fpn = corresponding_boxes[torch.logical_and(pos_indices < grid_total[fpn_idx+1],pos_indices >= grid_total[fpn_idx])]
            
            corresponding_boxes_fpn = gt_boxes[corresponding_boxes_fpn]
            corresponding_anchors_fpn = anchors_stacked_all[pos_indices_fpn_all]
            
            transformed_boxes = normalize_box(corresponding_anchors_fpn, corresponding_boxes_fpn).float()

            if len(pos_indices_fpn) > 0:
                ground_coord_one[4*idx[0], idx[1], idx[2]] = transformed_boxes[:,0]
                ground_coord_one[4*idx[0] + 1, idx[1], idx[2]] = transformed_boxes[:,1]
                ground_coord_one[4*idx[0] + 2, idx[1], idx[2]] = transformed_boxes[:,2]
                ground_coord_one[4*idx[0] + 3, idx[1], idx[2]] = transformed_boxes[:,3]

            ground_class.append(ground_class_one)
            ground_coord.append(ground_coord_one)
            
            del ground_class_one, ground_coord_one
            del pos_indices_fpn_all, pos_indices_fpn, neg_indices_fpn_all, neg_indices_fpn
            del idx_neg, idx, corresponding_boxes_fpn, corresponding_anchors_fpn, transformed_boxes
            
        del anchors_stacked_all, ious, pos_indices, neg_indices, cross_boundary, gt_boxes, corresponding_boxes, max_indices
        #####################################################
        self.ground_dict[key] = (ground_class, ground_coord)
        
        return ground_class, ground_coord

    


    def create_ground_truth(self, bboxes, index, anchor_list, grid_size_list, image_size):
        # This function create the ground truth for one image for all the FPN levels
        # It also caches the ground truth for the image using its index
        # Input:
        #       bboxes:      (n_boxes,4)
        #       index:       scalar (the index of the image in the total dataset)
        #       grid_size:   list:len(FPN){tuple:len(2)}
        #       anchor_list: list:len(FPN){(num_anchors, grid_size[0], grid_size[1], 4)}
        # Output:
        #       ground_clas_list: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
        #       ground_coord_list: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}

        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        # TODO create ground truth for a single image

        ground_coord_list = []
        ground_clas_list = []

        # Converting bounding boxes to x_c, y_c, width, height
        conv_bboxes = conv_box_to_xywh(bboxes)

        for k in range(5):
            anchors = anchor_list[k].to(device)
            grid_size = grid_size_list[k]

            all_ground_clas = (-1) * torch.ones(self.num_anchors, conv_bboxes.shape[0], grid_size[0], grid_size[1]).to(device)
            ground_clas = (-1) * torch.ones(self.num_anchors, grid_size[0], grid_size[1])
            ground_coord = torch.ones(self.num_anchors, 4, grid_size[0], grid_size[1])
            temp_coord = torch.ones(self.num_anchors, 4, grid_size[0], grid_size[1]).to(device)


            for i in range(self.num_anchors):

                # Removing cross boundary anchors
                idx_cross_boundary = torch.where((anchors[i,:,:,0] + (anchors[i,:,:,2]/2) > image_size[1]) | 
                                    (anchors[i,:,:,0] - (anchors[i,:,:,2]/2) < 0) | 
                                    (anchors[i,:,:,1] + (anchors[i,:,:,3]/2) > image_size[0]) | 
                                    (anchors[i,:,:,1] - (anchors[i,:,:,3]/2) < 0))


                all_ground_clas[i, :, idx_cross_boundary[0], idx_cross_boundary[1]] = 5.

                idx_other_x, idx_other_y = torch.where(all_ground_clas[i,0] != 5.)
                # idx_other_x = idx_other_x.to(device)
                # idx_other_y = idx_other_y.to(device)


                remain_anchors = anchors[i, idx_other_x, idx_other_y, :].reshape(-1,4)

                all_ground_clas = torch.where(all_ground_clas[i] == 5., -1., all_ground_clas)

                iou_mat = IOU_gt(remain_anchors, bboxes)
            
                max_iou_each_bbox = torch.max(iou_mat, dim=0)[0]

                for j in range(bboxes.shape[0]):
                    pos_res_bools = torch.logical_or(iou_mat[:,j] >= 0.99*max_iou_each_bbox[j].item(), iou_mat[:,j] > 0.7)
                    pos_res_idx = torch.where(pos_res_bools == True)[0]
                    all_ground_clas[i, j, idx_other_x[pos_res_idx.long()], idx_other_y[pos_res_idx.long()]] = 2.
                    neg_res_idx = torch.where(torch.logical_and(iou_mat[:,j] < 0.3, ~pos_res_bools) == True)[0]
                    all_ground_clas[i, j, idx_other_x[neg_res_idx.long()], idx_other_y[neg_res_idx.long()]] = 0.

                idx_check = torch.sum(all_ground_clas[i], dim=0)
                pos_idx_combined = idx_check > 0.
                neg_idx_combined = idx_check == 0.
                rest_idx_combined = idx_check < 0.

                ground_clas[i,rest_idx_combined] = -1.
                ground_clas[i,pos_idx_combined]   = 1.
                ground_clas[i,neg_idx_combined]   = 0.

                positive_label_indices = torch.where(ground_clas[i] == 1.)
        
                positive_anchors = anchors[i, positive_label_indices[0], positive_label_indices[1], :].reshape(-1,4)
                iou_mat_pos_anc = IOU_gt(positive_anchors, bboxes)

                idx_max_iou_each_grid = torch.max(iou_mat_pos_anc, dim=1)[1]
                final_conv_bboxes = torch.stack([conv_bboxes[idx] for idx in idx_max_iou_each_grid]).float().to(device)

                temp_coord[i, :, positive_label_indices[0], positive_label_indices[1]] = final_conv_bboxes.T

                ground_coord[i,0,:,:] = (temp_coord[i,0,:,:] - anchors[i,:,:,0]) / anchors[i,:,:,2]    # t_x
                ground_coord[i,1,:,:] = (temp_coord[i,1,:,:] - anchors[i,:,:,1]) / anchors[i,:,:,3]    # t_y
                ground_coord[i,2,:,:] = torch.log(temp_coord[i,2,:,:] / anchors[i,:,:,2])              # t_w
                ground_coord[i,3,:,:] = torch.log(temp_coord[i,3,:,:] / anchors[i,:,:,3])              # t_h
            
            ground_coord = ground_coord.reshape(4*self.num_anchors, grid_size[0], grid_size[1])

            ground_coord_list.append(ground_coord)
            ground_clas_list.append(ground_clas)
      
        self.ground_dict[key] = (ground_clas_list, ground_coord_list)
        return ground_clas_list, ground_coord_list


    def loss_class(self, p_out, n_out):
        # Compute the loss of the classifier
        # Input:
        #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
        #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels

        # torch.nn.BCELoss()
        # TODO compute classifier's loss
        p_tar = torch.ones_like(p_out)
        n_tar = torch.zeros_like(n_out)

        out = torch.cat((p_out, n_out), dim=0)
        tar = torch.cat((p_tar, n_tar), dim=0)

        BCE_obj = torch.nn.BCELoss(reduction='mean')
        loss = BCE_obj(out, tar)

        return loss


    def loss_reg(self, pos_target_coord, pos_out_r):
        # Compute the loss of the regressor
        # Input:
        #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
        #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
        # torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss

        pos_out_r = pos_out_r.to(device)
        pos_target_coord = pos_target_coord.to(device)
        L1_loss_obj = torch.nn.SmoothL1Loss(reduction='mean')
        loss = L1_loss_obj(pos_out_r, pos_target_coord)

        return loss


    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=1, effective_batch=150):
        # Compute the total loss for the FPN heads
        # Input:
        #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
        #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
        #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
        #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
        #       l: weighting lambda between the two losses
        # Output:
        #       loss: scalar
        #       loss_c: scalar
        #       loss_r: scalar
        
        loss = 0
        loss_c = 0
        loss_r = 0

        for each_level in zip(clas_out_list, regr_out_list, targ_clas_list, targ_regr_list):

            clas_out = each_level[0]
            regr_out = each_level[1]
            targ_clas = each_level[2]
            targ_regr = each_level[3]

            clas_out_all = clas_out.flatten()
            regr_out_all = regr_out.permute(0,2,3,1).reshape(-1,4)

            targ_clas_all = targ_clas.flatten()
            targ_regr_all = targ_regr.permute(0,2,3,1).reshape(-1,4)
            
            pos_mask = targ_clas_all == 1.              # Mask for positive indices
            neg_mask = targ_clas_all == 0.              # Mask for positive indices

            num_pos_ind = pos_mask.sum().item()        # No of positive indices
            num_neg_ind = neg_mask.sum().item()        # No of negative indices

            if num_pos_ind > (effective_batch/2):
                rand_pos_idx = torch.randperm(num_pos_ind)[:int(effective_batch/2)]
                rand_neg_idx = torch.randperm(num_neg_ind)[:int(effective_batch/2)]
                pos_clas = clas_out_all[pos_mask][rand_pos_idx]
                neg_clas = clas_out_all[neg_mask][rand_neg_idx]
                regr_pred = regr_out_all[pos_mask][rand_pos_idx]
                regr_tar  = targ_regr_all[pos_mask][rand_pos_idx]
            else:

                rand_neg_idx = torch.randperm(num_neg_ind)
                pos_clas = clas_out_all[pos_mask]
                neg_clas = clas_out_all[neg_mask][rand_neg_idx][:(effective_batch - num_pos_ind)]
                regr_pred = regr_out_all[pos_mask]
                regr_tar  = targ_regr_all[pos_mask]
            
            loss += loss_c + l*loss_r
            loss_c += self.loss_class(pos_clas, neg_clas)
            loss_r += self.loss_reg(regr_tar, regr_pred)

        print("loss tot : ", loss)
        print("loss clas : ", loss_c)
        print("loss reg : ", loss_r)

        return loss, loss_c, loss_r

    
    def training_step(self, batch, batch_idx):
        images=batch['images']
        index=batch['index']
        bounding_boxes=batch['bboxes']
        with torch.no_grad():
          backbone = self.resnet50_fpn(images)
        backbone_list = [backbone["0"], backbone["1"], backbone["2"], backbone["3"], backbone["pool"]]
        class_logits, box_pred = self.forward(backbone_list)

        gt_labels, regressor_target = self.create_batch_truth(bounding_boxes,index,images.shape[-2:])

        loss, loss_class, loss_regr = self.compute_loss(class_logits, box_pred, gt_labels, regressor_target, l=0.2, effective_batch=150)
    
        del images, bounding_boxes, index
        del gt_labels, regressor_target
        torch.cuda.empty_cache()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("loss_class", loss_class, prog_bar=True)
        self.log("loss_regr", loss_regr, prog_bar=True)

        return {"loss":loss, "classifier_loss":loss_class, "regressor_loss":loss_regr}

    def training_epoch_end(self, training_step_outputs):
        avg_train_loss = 0
        avg_class_loss = 0
        avg_reg_loss = 0
        for i in range(len(training_step_outputs)):
            avg_train_loss += training_step_outputs[i]["loss"].detach().cpu().item()
            avg_class_loss += training_step_outputs[i]["classifier_loss"].detach().cpu().item()
            avg_reg_loss += training_step_outputs[i]["regressor_loss"].detach().cpu().item()

        self.train_loss_epoch.append(avg_train_loss)
        self.train_class_loss_epoch.append(avg_class_loss)
        self.train_reg_loss_epoch.append(avg_reg_loss)


    def validation_step(self, batch, batch_idx):
        images=batch['images']
        index=batch['index']
        bounding_boxes=batch['bboxes']
        with torch.no_grad():
          backbone = self.resnet50_fpn(images)
        backbone_list = [backbone["0"], backbone["1"], backbone["2"], backbone["3"], backbone["pool"]]
        class_logits, box_pred = self.forward(backbone_list)


        gt_labels, regressor_target = self.create_batch_truth(bounding_boxes,index,images.shape[-2:])

        val_loss, loss_class, loss_regr = self.compute_loss(class_logits, box_pred, gt_labels, regressor_target, l=0.2, effective_batch=150)

        del images, bounding_boxes, index
        del gt_labels, regressor_target
        torch.cuda.empty_cache()

        self.log("val_loss", val_loss)
        return {"loss":val_loss, "classifier_loss":loss_class, "regressor_loss":loss_regr}

    def validation_epoch_end(self, outputs):
        avg_train_loss = 0
        avg_class_loss = 0
        avg_reg_loss = 0
        for i in range(len(outputs)):
            avg_train_loss += outputs[i]["loss"].detach().cpu().item()
            avg_class_loss += outputs[i]["classifier_loss"].detach().cpu().item()
            avg_reg_loss += outputs[i]["regressor_loss"].detach().cpu().item()
        
        self.val_loss_epoch.append(avg_train_loss)
        self.val_class_loss_epoch.append(avg_class_loss)
        self.val_reg_loss_epoch.append(avg_reg_loss)


    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),lr = 0.002,weight_decay=1.0e-4,momentum=0.90)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,35], gamma=0.20)

        return optimizer 


    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=2000, keep_num_postNMS=1000):
        # Post process for the outputs for a batch of images
        # Input:
        #       out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
        #       out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
        #       IOU_thresh: scalar that is the IOU threshold for the NMS
        #       keep_num_preNMS: number of masks we will keep from each image before the NMS
        #       keep_num_postNMS: number of masks we will keep from each image after the NMS
        # Output:
        #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
        #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
        nms_clas_list = []
        nms_prebox_list = []
        bz = out_c[0].shape[0]
        for i in range(bz):
            one_out_c = [out_c[j][i].unsqueeze(0) for j in range(5)]
            one_out_r = [out_r[j][i].unsqueeze(0) for j in range(5)]
            one_nms_clas, one_nms_coord = self.postprocessImg(one_out_c, one_out_r, IOU_thresh, keep_num_preNMS, keep_num_postNMS)
            nms_clas_list.append(one_nms_clas)
            nms_prebox_list.append(one_nms_coord)
            
        return nms_clas_list, nms_prebox_list


    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        # Post process the output for one image
        # Input:
        #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
        #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
        # Output:
        #       nms_clas: (Post_NMS_boxes)
        #       nms_prebox: (Post_NMS_boxes,4)
        # print("Inside post process class: ", len(mat_clas))
        # print("Inside post process coord : ", mat_coord)

        anchors_list = self.get_anchors()
        flatten_regr_all, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, anchors_list)
        decoded_coord = output_decoding(flatten_regr_all, flatten_anchors)

        idx_cross_boundary = torch.logical_or((decoded_coord[:,2] > self.image_size[1]), 
                    torch.logical_or((decoded_coord[:,0] < 0), 
                    torch.logical_or((decoded_coord[:,3] > self.image_size[0]), (decoded_coord[:,1] < 0))))

        remaining_coord_idx = torch.where(idx_cross_boundary == False)
    
        remaining_coord = decoded_coord[remaining_coord_idx]
        remaining_clas = flatten_clas[remaining_coord_idx]

        sorted_clas, sorted_clas_idx = torch.sort(remaining_clas, descending=True)
        sorted_coord = remaining_coord[sorted_clas_idx]

        top_pre_nms_clas = sorted_clas[:keep_num_preNMS]
        top_pre_nms_coord = sorted_coord[:keep_num_preNMS]

        top_post_nms_clas, top_post_nms_coord = self.NMS(top_pre_nms_clas, top_pre_nms_coord, IOU_thresh)

        nms_clas = top_post_nms_clas[:keep_num_postNMS]
        nms_prebox = top_post_nms_coord[:keep_num_postNMS]

        return nms_clas, nms_prebox



    def NMS(self, clas, prebox, thresh):
        # Input:
        #       clas: (top_k_boxes) (scores of the top k boxes)
        #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        # Output:
        #       nms_clas: (Post_NMS_boxes)
        #       nms_prebox: (Post_NMS_boxes,4)
        
        gauss_sigma = 0.5
        # conv_prebox = conv_box_to_corners(prebox)
        conv_prebox = prebox

        for i in range(1):

            ious = torchvision.ops.box_iou(conv_prebox, conv_prebox).triu(diagonal=1)
            ious[ious < thresh] = 0
            ious_cmax = ious.max(0)[0].expand(clas.shape[0], clas.shape[0]).T
            
            decay = torch.exp(-(ious * 2 - ious_cmax * 2) / gauss_sigma)

            decay = decay.min(dim=0)[0]
            decayed_clas = decay * clas
            sorted_decayed_clas, sorted_decayed_idx = torch.sort(decayed_clas, descending=True)
            nms_prebox = conv_prebox[sorted_decayed_idx]
            nms_clas = sorted_decayed_clas

            conv_prebox = nms_prebox
            clas = nms_clas
       
        return nms_clas, nms_prebox


if __name__ == "__main__":
    rpn_obj = RPNHead()
    # bboxes = torch.tensor(np.array([[167., 100.16265, 312.80722, 204.6747], [119.421684, 57.391567, 232.48193, 190.36748 ]]))
    bboxes = torch.tensor(np.array([[511.3333, 338.8949, 762.2500, 562.6717],[548.5833, 237.4750, 777.7999, 531.3412]]))
    index = 2
    image_size = (800, 1088)
    # anchors = rpn_obj.create_anchors(anchors_param["ratio"], anchors_param["scale"], anchors_param["grid_size"], anchors_param["stride"])



