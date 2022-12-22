import torch
import torch.nn.functional as F
from torch import nn
from utils import *
from BoxHead import BoxHead

class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14, batch_size = 4):

        self.C=Classes
        self.P=P
        self.keep_topK = 100
        self.batch_size = batch_size
        self.image_size = (800,1088)
        self.pretrained_boxhead_path = '/content/drive/MyDrive/CIS 6800/SOLO dataset/model_trained_boxhead_2.pth'
        self.pretrained_fpn_path = '/content/drive/MyDrive/CIS 6800/SOLO dataset/checkpoint680.pth'
        self.model_boxhead = BoxHead()
        self.model_boxhead.load_state_dict(torch.load(self.pretrained_boxhead_path, map_location=device))
        with torch.no_grad():
          self.backbone, self.rpn = pretrained_models_680(self.pretrained_fpn_path)
        self.train_loss_epoch = []


        self.val_loss_epoch = []
        # TODO initialize MaskHead

        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )

    def preprocess_ground_truth_creation(self, proposals, class_logits, box_regression, gt_labels,bbox ,masks , IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=10):
        # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
        # and create the ground truth for the Mask Head
        #
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
        #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,C,2*P,2*P)}
        num_proposals = proposals[0].shape[0]
        boxes = []
        scores = []
        labels = []
        gt_masks = []
        for i, each_proposal in enumerate(proposals):
            
            each_proposal = each_proposal.to(device)
            box_regression = box_regression.to(device)
            class_logits = class_logits.to(device)
            # gt_labels[i] = gt_labels[i] - 1
            one_image_boxes = box_regression[i*num_proposals:(i+1)*num_proposals]          # Shape (num_proposals, 12)
            one_image_logits = class_logits[i*num_proposals:(i+1)*num_proposals]           # Shape (num_proposals, 4)
            one_image_scores, one_image_label = torch.max(one_image_logits, dim=1)
            one_image_label = one_image_label.clone().int() - 1
            non_bg_label_idx = torch.where(one_image_label >= 0)[0]

            if len(non_bg_label_idx) != 0: 
                class_labels = one_image_label[non_bg_label_idx]
                all_class_boxes = one_image_boxes[non_bg_label_idx]
                class_boxes =  torch.stack([all_class_boxes[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])      # Shape(filtered_labels, 4) ([t_x,t_y,t_w,t_h])
                decoded_boxes = output_decoding_postprocess(class_boxes, each_proposal[non_bg_label_idx])                          # (x1,y1,x2,y2)
                
                valid_boxes_idx = torch.where((decoded_boxes[:,0] >= 0) & (decoded_boxes[:,2] < 1088) & (decoded_boxes[:,1] > 0) & (decoded_boxes[:,3] < 800))
                valid_boxes = decoded_boxes[valid_boxes_idx]
                valid_clases = one_image_label[non_bg_label_idx][valid_boxes_idx]
                valid_scores = one_image_scores[non_bg_label_idx][valid_boxes_idx]
                sorted_scores_pre_nms, sorted_idx = torch.sort(valid_scores, descending=True)
                sorted_clases_pre_nms = valid_clases[sorted_idx]
                sorted_boxes_pre_nms = valid_boxes[sorted_idx]

                iou_check = torchvision.ops.box_iou(sorted_boxes_pre_nms.to(device), bbox[i].to(device))
                iou_idx = (iou_check > 0.5).nonzero()
                above_thres_idx = iou_idx[:,0]
                above_thres_gt = iou_idx[:,1]


                masks_gt_all = masks[i][above_thres_gt.cpu()]
                                
                sorted_boxes_pre_nms = sorted_boxes_pre_nms[above_thres_idx.cpu()]
                sorted_clases_pre_nms = sorted_clases_pre_nms[above_thres_idx.cpu()]
                sorted_scores_pre_nms = sorted_scores_pre_nms[above_thres_idx.cpu()]

                if len(sorted_clases_pre_nms) > keep_num_preNMS:
                    clases_pre_nms = sorted_clases_pre_nms[:keep_num_preNMS]
                    boxes_pre_nms = sorted_boxes_pre_nms[:keep_num_preNMS]
                    scores_pre_nms = sorted_scores_pre_nms[:keep_num_preNMS]
                    masks_pre_nms = masks_gt_all[:keep_num_preNMS]
                else:
                    clases_pre_nms = sorted_clases_pre_nms
                    boxes_pre_nms = sorted_boxes_pre_nms
                    scores_pre_nms = sorted_scores_pre_nms
                    masks_pre_nms = masks_gt_all

                clases_post_nms, scores_post_nms, boxes_post_nms, masks_post_nms = self.nms_preprocess_gt(clases_pre_nms, boxes_pre_nms, scores_pre_nms, masks_pre_nms, IOU_thres=IOU_thresh, keep_num_postNMS=keep_num_postNMS)

                gt_mask_one = torch.zeros(clases_post_nms.shape[0],self.image_size[0], self.image_size[1]).to(device)

                for j in range(clases_post_nms.shape[0]):
                    gt_mask_one[j,boxes_post_nms[j,1].int():boxes_post_nms[j,3].int(),boxes_post_nms[j,0].int():boxes_post_nms[j,2].int()] = 1
                gt_mask_one = gt_mask_one * masks_post_nms
                gt_mask_one = F.interpolate(gt_mask_one.unsqueeze(0), size=(2*self.P,2*self.P),mode='nearest').squeeze(0)
            
            gt_masks.append(gt_mask_one)
            boxes.append(boxes_post_nms)
            scores.append(scores_post_nms)
            labels.append(clases_post_nms)

        return boxes, scores, labels, gt_masks

    def nms_preprocess_gt(self,clases,boxes,scores, masks, IOU_thres=0.5, keep_num_postNMS=10):
        # Input:
        #       clases: (num_preNMS, )
        #       boxes:  (num_preNMS, 4)
        #       scores: (num_preNMS,)
        # Output:
        #       boxes:  (post_NMS_boxes_per_image,4) ([x1,y1,x2,y2] format)
        #       scores: (post_NMS_boxes_per_image)   ( the score for the top class for the regressed box)
        #       labels: (post_NMS_boxes_per_image)  (top category of each regressed box)
        ###########################################################
        boxes = boxes.to(device)
        clases = clases.to(device)
        scores = scores.to(device)
        masks = masks.to(device)
        scores_all = [[],[],[]]
        boxes_all = [[],[],[]]
        clas_all = [[],[],[]]
        masks_all = [[],[],[]]

        for i in range(3):
            each_label_idx = torch.where(clases == i)[0]
            if len(each_label_idx) == 0:
              continue
            each_clas_boxes = boxes[each_label_idx]
            each_clas_score = scores[each_label_idx]
            each_clas_mask = masks[each_label_idx]


            start_x_torched = each_clas_boxes[:, 0]
            start_y_torched = each_clas_boxes[:, 1]
            end_x_torched   = each_clas_boxes[:, 2]
            end_y_torched   = each_clas_boxes[:, 3]

            areas_torched = (end_x_torched - start_x_torched + 1) * (end_y_torched - start_y_torched + 1)

            order_torched = torch.argsort(each_clas_score)

            while len(order_torched) > 0:
                # The index of largest confidence score
                index = order_torched[-1]
                
                # Pick the bounding box with largest confidence score
                boxes_all[i].append(each_clas_boxes[index].detach())
                scores_all[i].append(each_clas_score[index].detach())
                masks_all[i].append(each_clas_mask[index].detach())

                if len(boxes_all[i]) == keep_num_postNMS:
                    break

                # Compute ordinates of intersection-over-union(IOU)
                x1 = torch.maximum(start_x_torched[index], start_x_torched[order_torched[:-1]]).to(device)
                x2 = torch.minimum(end_x_torched[index], end_x_torched[order_torched[:-1]]).to(device)
                y1 = torch.maximum(start_y_torched[index], start_y_torched[order_torched[:-1]]).to(device)
                y2 = torch.minimum(end_y_torched[index], end_y_torched[order_torched[:-1]]).to(device)

                # Compute areas of intersection-over-union
                w = torch.maximum(torch.tensor([0]).to(device), x2 - x1 + 1)
                h = torch.maximum(torch.tensor([0]).to(device), y2 - y1 + 1)
                intersection = w * h

                # Compute the ratio between intersection and union
                ratio = intersection / (areas_torched[index] + areas_torched[order_torched[:-1]] - intersection)
                left = torch.where(ratio < IOU_thres)[0]
                order_torched = order_torched[left]
            clas_all[i] = [i]*len(scores_all[i])
        
        fin_masks = torch.cat([torch.stack(one_mask) for one_mask in masks_all if len(one_mask)!=0]).reshape(-1,800,1088)
        fin_scores = torch.cat([torch.tensor(one_score).reshape(-1,1) for one_score in scores_all if len(one_score)!=0],dim=0).reshape(-1,1)
        fin_boxes = torch.cat([torch.stack(one_box) for one_box in boxes_all if len(one_box)!=0]).reshape(-1,4)
        fin_clas = torch.cat([torch.tensor(one_clas) for one_clas in clas_all if len(one_clas)!=0]).reshape(-1,1)
        return fin_clas, fin_scores, fin_boxes, fin_masks

    def MultiScaleRoiAlign_maskhead(self, fpn_feat_list,proposals):
        #####################################
        # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        # Input:
        #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
        #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
        #      P: scalar
        # Output:
        #      feature_vectors: len(bz) (total_proposals, 256, 2*P, 2*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        roi_alls = []
        for i in range(len(proposals)):
            each_proposal= conv_box_to_xywh(proposals[i])
            k = (torch.log2(torch.sqrt(each_proposal[:,2]*each_proposal[:,3])/224.) + 4).int()
            k = torch.clamp(k, min=2., max=5.).int()
            each_proposal = conv_box_to_corners(each_proposal)

            scaling_vals = torch.pow(2,k).reshape(-1,1)*torch.ones_like(each_proposal)
            scaled_proposals = each_proposal / scaling_vals

            fpn_list_each_proposal = [fpn_feat_list[j][i].unsqueeze(0) for j in range(5)]
            roi_vals = torch.stack([torchvision.ops.roi_align(fpn_list_each_proposal[k[n]-2], [scaled_proposals[n].view(1,4)], (self.P,self.P)).squeeze(0) for n in range(k.shape[0])], dim=0)
            roi_alls.append(roi_vals)

        return roi_alls

    # general function that takes the input list of tensors and concatenates them along the first tensor dimension
    # Input:
    #      input_list: list:len(bz){(dim1,?)}
    # Output:
    #      output_tensor: (sum_of_dim1,?)
    def flatten_inputs(self,input_list):
        output_tensor = torch.cat(input_list, dim=0)
        return output_tensor


    def postprocess_mask(self, masks_outputs, labels):
        # This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
        # back to the original image size
        # Use the regressed boxes to distinguish between the images
        # Input:
        #       masks_outputs: (total_boxes,C,2*P,2*P)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
        # Output:
        #       projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
        projected_masks = []
        count = 0
        for i, each_label in enumerate(labels):
            each_masks = masks_outputs[count:(count+each_label.shape[0])]
            fin_each_masks = torch.stack([each_masks[l,one_each_label.item()] for l,one_each_label in enumerate(each_label)])
            count += each_label.shape[0]
            one_projected_mask = F.interpolate(fin_each_masks.unsqueeze(0), size=(800,1088), mode="bilinear").squeeze(0)
            one_projected_mask[one_projected_mask >= 0.5] = 1
            one_projected_mask[one_projected_mask < 0.5] = 0
            projected_masks.append(one_projected_mask)     

        return projected_masks


    def compute_loss(self,mask_output,labels,gt_masks):
        # Compute the total loss of the Mask Head
        # Input:
        #      mask_output: (total_boxes,C,2*P,2*P)
        #      labels: len(bz)(total_boxes)
        #      gt_masks: len(bz) (total_boxes,2*P,2*P)
        # Output:
        #      mask_loss
        flattened_gt_masks = self.flatten_inputs(gt_masks)
        flattened_labels = self.flatten_inputs(labels)

        mask_target = []
        for i in range(len(flattened_labels)):
            one_mask_output = mask_output[i]
            mask_target.append(one_mask_output[flattened_labels[i].item()])
        mask_target = torch.stack(mask_target)
        
        criterion = nn.BCELoss()

        mask_loss = criterion(mask_target, flattened_gt_masks)

        return mask_loss.mean()

    
    def forward(self,  feature_boxes):
        # Forward the pooled feature map Mask Head
        # Input:
        #        features: (total_boxes, 256,P,P)
        # Outputs:
        #        mask_outputs: (total_boxes,C,2*P,2*P)

        mask_outputs = self.mask_head(feature_boxes)

        return mask_outputs

if __name__ == '__main__':
    print("running")
