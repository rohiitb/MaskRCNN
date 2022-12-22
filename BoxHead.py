import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from utils import *
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead,self).__init__()

        self.C=Classes
        self.P=P
        self.image_size = (800,1088)

        # TODO initialize BoxHead
        self.intermediate_layer = nn.Sequential(
            nn.Linear(in_features=256*self.P*self.P, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        )

        self.classfier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=self.C+1),
            nn.Softmax(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4*self.C)
       )


    def create_ground_truth(self,proposals,gt_labels,bbox):
        """     
          This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
          Input:
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            gt_labels: list:len(bz) {(n_obj)}
            bbox: list:len(bz){(n_obj, 4)}
          Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals,1) (the class that the proposal is assigned)
            regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
        """        
        all_labels = []
        all_regressor_targets = []
        for per_image_proposal, per_image_gt_label, per_image_bbox in zip(proposals, gt_labels, bbox):
            per_image_gt_label = per_image_gt_label.clone().detach().to(device).float()
            each_labels = (-1.)*torch.ones(per_image_proposal.shape[0]).float().to(device)
            each_regressor_target = torch.ones(per_image_proposal.shape[0], 4).to(device)
            per_image_bbox = per_image_bbox.to(device)
            iou = torchvision.ops.box_iou(per_image_proposal, per_image_bbox)
            per_image_proposal = conv_box_to_xywh(per_image_proposal)
            per_image_bbox = conv_box_to_xywh(per_image_bbox)

            max_iou, max_iou_idx = torch.max(iou, dim=1)
            max_iou_idx = max_iou_idx.long()
            above_thres = torch.where(max_iou > 0.5)

            if len(above_thres[0]) != 0. : 
                each_labels[above_thres[0].long()] = torch.stack([per_image_gt_label[i] for i in max_iou_idx[above_thres[0].long()]])
                # print("Inside loss iteration : ", torch.where(each_labels < 0.)[0].shape)
                each_regressor_target[above_thres[0].long()] = torch.stack([per_image_bbox[i] for i in max_iou_idx[above_thres[0].long()]])

                conv_each_regressor_target = torch.zeros_like(per_image_proposal)
                conv_each_regressor_target[:,0] = (each_regressor_target[:,0] - per_image_proposal[:,0]) / per_image_proposal[:,2]
                conv_each_regressor_target[:,1] = (each_regressor_target[:,1] - per_image_proposal[:,1]) / per_image_proposal[:,3]
                conv_each_regressor_target[:,2] = torch.log(each_regressor_target[:,2]/per_image_proposal[:,2])
                conv_each_regressor_target[:,3] = torch.log(each_regressor_target[:,3]/per_image_proposal[:,3])

                all_labels.append(each_labels)
                all_regressor_targets.append(conv_each_regressor_target)
            else : 
                # print("Entering inside")
                all_labels.append(each_labels)
                all_regressor_targets.append(each_regressor_target)


        labels = torch.cat(all_labels, dim=0)
        background_mask = labels < 0.
        regressor_target = torch.cat(all_regressor_targets, dim=0)
        regressor_target[background_mask] = 0.
        return labels,regressor_target



    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
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
            roi_vals = torch.stack([torchvision.ops.roi_align(fpn_list_each_proposal[k[n]-2], [scaled_proposals[n].view(1,4)], (P,P)).squeeze(0) for n in range(k.shape[0])], dim=0).reshape(-1,256*P*P)
            roi_alls.append(roi_vals)
        feature_vectors = torch.cat(roi_alls, dim=0)

        return feature_vectors


    def Matrix_NMS(self,clas,prebox,labels, thresh, method='gauss', gauss_sigma=0.5):
    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes) in form (x,y,w,h)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
        ##################################
        # TODO perform NM

        ious = torchvision.ops.box_iou(prebox, prebox).triu(diagonal=1)
        ious[ious < thresh] = 0
        ious_cmax = ious.max(0)[0].expand(clas.shape[0], clas.shape[0]).T
        
        if method == 'gauss':
            decay = torch.exp(-(ious * 2 - ious_cmax * 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        decayed_clas = decay * clas
        sorted_decayed_clas, sorted_decayed_idx = torch.sort(decayed_clas, descending=True)
        nms_prebox = prebox[sorted_decayed_idx]
        nms_clas = sorted_decayed_clas
        nms_label = labels[sorted_decayed_idx]

        ##################################
        return nms_clas, nms_prebox, nms_label
    

    def postprocess_detections_pre(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=20):
    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        class_logits = class_logits.cpu()
        box_regression = box_regression.cpu()


        class_scores, class_labels = torch.max(class_logits, dim=1)
        class_scores = torch.where(class_labels == 0., 0., class_scores)


        class_labels_cor = class_labels.clone() - 1.
        class_labels_cor = torch.where(class_labels_cor < 0., 0., class_labels_cor).int()
        
        filtered_box_regression = torch.stack([one_box_regression[4*(class_labels_cor[i]):4*class_labels_cor[i]+4] for i, one_box_regression in enumerate(box_regression)])

        boxes = []
        scores = []
        labels = []
        count = 0
        for i, each_proposal in enumerate(proposals):
            each_proposal = each_proposal.clone().detach().to(device)
            # print("Each proposals shape : ", i, each_proposal.shape)
            unsorted_box_regression = filtered_box_regression[count:(i+1)*each_proposal.shape[0]].to(device)
            unsorted_class_scores = class_scores[count:(i+1)*each_proposal.shape[0]]
            unsorted_class_labels= class_labels_cor[count:(i+1)*each_proposal.shape[0]]
            # print("Unsorted box reg : ", i,unsorted_box_regression.shape)
            # print("Unsorted class score : ", i, unsorted_class_scores.shape)
            # print(i, unsorted_class_idx)

            conv_unsorted_box_regression = output_decoding(unsorted_box_regression, each_proposal)            # Convert to x1,y1,x2,y2
            # print("Decoded : ", conv_unsorted_box_regression.shape)

            non_background_idx = torch.where(unsorted_class_scores != 0.)
            # print("background idx : ", non_background_idx[0].shape)
            # print("background : ", non_background_idx)
            conv_unsorted_box_regression = conv_unsorted_box_regression[non_background_idx]
            unsorted_class_scores = unsorted_class_scores[non_background_idx]
            unsorted_class_labels = unsorted_class_labels[non_background_idx]

            idx_cross_boundary = torch.logical_or(conv_unsorted_box_regression[:,2] >= self.image_size[1], 
                                torch.logical_or(conv_unsorted_box_regression[:,0] < 0, 
                                torch.logical_or(conv_unsorted_box_regression[:,3] >= self.image_size[0], conv_unsorted_box_regression[:,1] < 0)))
            
            remaining_box_reg_idx = torch.where(idx_cross_boundary == False)
            # print("remaining box reg : ", remaining_box_reg_idx[0].shape[0])

            
            # if remaining_box_reg_idx[0].shape[0] == 0:
            #   remaining_box_reg = conv_unsorted_box_regression
            #   remaining_class_scores =   unsorted_class_scores
            #   remaining_class_labels =   unsorted_class_labels
            # else: 
            remaining_box_reg = conv_unsorted_box_regression[remaining_box_reg_idx]
            remaining_class_scores = unsorted_class_scores[remaining_box_reg_idx]
            remaining_class_labels = unsorted_class_labels[remaining_box_reg_idx]


            sorted_class_scores, sorted_class_idx = torch.sort(remaining_class_scores, descending=True)
            sorted_box_reg = remaining_box_reg[sorted_class_idx]
            sorted_class_labels = remaining_class_labels[sorted_class_idx]

            if sorted_class_scores.shape[0] > keep_num_preNMS:
                sorted_class_scores = sorted_class_scores[:keep_num_preNMS]
                sorted_box_reg = sorted_box_reg[:keep_num_preNMS]
                sorted_class_labels = sorted_class_labels[:keep_num_preNMS]

            # print("Last : ", sorted_class_scores)

            each_image_box = []
            each_image_class_scores = []
            each_image_class_labels = []
            for i in range(3):
                sorted_i_idx = torch.where(sorted_class_labels == i)
                if len(sorted_i_idx[0]) == 0:
                  continue
                sorted_class_scores_i = sorted_class_scores[sorted_i_idx]
                sorted_box_reg_i = sorted_box_reg[sorted_i_idx]
                sorted_class_labels_i = sorted_class_labels[sorted_i_idx]
                # print("Before sorted class labesl : ", sorted_class_labels_i.shape)
                # print("Before sorted box reg : ", sorted_box_reg_i.shape)
                # print("Before sorted class scores : ", sorted_class_scores_i.shape)
                post_nms_class_scores, post_nms_box_reg, post_nms_class_labels = self.NMS(sorted_class_scores_i, sorted_box_reg_i, sorted_class_labels_i, thresh=conf_thresh)
                # print("After sorted class labesl : ", post_nms_class_scores.shape)
                # print("After sorted box reg : ", post_nms_box_reg.shape)
                # print("After sorted class scores : ", post_nms_class_labels.shape)
                post_nms_class_scores = post_nms_class_scores[:keep_num_postNMS]
                post_nms_box_reg = post_nms_box_reg[:keep_num_postNMS]
                post_nms_class_labels = post_nms_class_labels[:keep_num_postNMS]
                # if post_nms_box_reg.shape[0]==0:
                #   continue
                each_image_box.append(post_nms_box_reg)
                each_image_class_scores.append(post_nms_class_scores)
                each_image_class_labels.append(post_nms_class_labels)

            # print("Score : ", each_image_class_scores)
            # print("Label : ", each_image_class_labels)
            # print("Box : ",            each_image_box)
            if len(each_image_class_labels) == 0:
              each_image_score = torch.tensor(each_image_class_scores)
              each_image_label = torch.tensor(each_image_class_labels)
              each_image_boxes = torch.tensor(each_image_box)
            else:  
              each_image_score = torch.cat(each_image_class_scores)
              each_image_label = torch.cat(each_image_class_labels)
              each_image_boxes = torch.cat(each_image_box, dim=0)

            # print("After Score : ", each_image_score.shape)
            # print("After Label : ", each_image_label.shape)
            # print("After Box : ",   each_image_boxes.shape)

            boxes.append(each_image_boxes)
            scores.append(each_image_score)
            labels.append(each_image_label)

            count += each_proposal.shape[0]

        return boxes, scores, labels


    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=20):
        # This function does the post processing for the results of the Box Head for a batch of images
        # Use the proposals to distinguish the outputs from each image
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        num_proposals = proposals[0].shape[0]
        boxes = []
        scores = []
        labels = []
        for i, each_proposal in enumerate(proposals):
            one_image_boxes = box_regression[i*num_proposals:(i+1)*num_proposals]          # Shape (num_proposals, 12)
            one_image_logits = class_logits[i*num_proposals:(i+1)*num_proposals]           # Shape (num_proposals, 4)
            one_image_scores, one_image_label = torch.max(one_image_logits, dim=1)
            one_image_label = one_image_label.clone().int() - 1
            non_bg_label_idx = torch.where(one_image_label >= 0)[0]

            if len(non_bg_label_idx) != 0: 
                class_labels = one_image_label[non_bg_label_idx]
                all_class_boxes = one_image_boxes[non_bg_label_idx]
                class_boxes =  torch.stack([all_class_boxes[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])      # Shape(filtered_labels, 4) ([t_x,t_y,t_w,t_h])
                decoded_boxes = output_decoding(class_boxes, each_proposal[non_bg_label_idx])                          # (x1,y1,x2,y2)
                
                valid_boxes_idx = torch.where((decoded_boxes[:,0] >= 0) & (decoded_boxes[:,2] < 1088) & (decoded_boxes[:,1] > 0) & (decoded_boxes[:,3] < 800))
                valid_boxes = decoded_boxes[valid_boxes_idx]
                valid_clases = one_image_label[non_bg_label_idx][valid_boxes_idx]
                valid_scores = one_image_scores[non_bg_label_idx][valid_boxes_idx]
                sorted_scores_pre_nms, sorted_idx = torch.sort(valid_scores, descending=True)
                sorted_clases_pre_nms = valid_clases[sorted_idx]
                sorted_boxes_pre_nms = valid_boxes[sorted_idx]

                if len(sorted_clases_pre_nms) > keep_num_preNMS:
                    clases_pre_nms = sorted_clases_pre_nms[:keep_num_preNMS]
                    boxes_pre_nms = sorted_boxes_pre_nms[:keep_num_preNMS]
                    scores_pre_nms = sorted_scores_pre_nms[:keep_num_preNMS]
                else:
                    clases_pre_nms = sorted_clases_pre_nms
                    boxes_pre_nms = sorted_boxes_pre_nms
                    scores_pre_nms = sorted_scores_pre_nms

                clases_post_nms, scores_post_nms, boxes_post_nms = self.greedy_nms(clases_pre_nms, boxes_pre_nms, scores_pre_nms, IOU_thres=conf_thresh, keep_num_postNMS=keep_num_postNMS)
        boxes.append(boxes_post_nms)
        scores.append(scores_post_nms)
        labels.append(clases_post_nms)

        return boxes, scores, labels


    def greedy_nms(self,clases,boxes,scores, IOU_thres=0.5, keep_num_postNMS=100):
        # Input:
        #       clases: (num_preNMS, )
        #       boxes:  (num_preNMS, 4)
        #       scores: (num_preNMS,)
        # Output:
        #       boxes:  (post_NMS_boxes_per_image,4) ([x1,y1,x2,y2] format)
        #       scores: (post_NMS_boxes_per_image)   ( the score for the top class for the regressed box)
        #       labels: (post_NMS_boxes_per_image)  (top category of each regressed box)
        ###########################################################

        scores_all = []
        boxes_all = []
        clas_all = []

        for i in range(3):
            scores_each = []
            boxes_each = []

            each_label_idx = torch.where(clases == i)[0]
            each_clas_boxes = boxes[each_label_idx]
            each_clas_score = scores[each_label_idx]

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
                boxes_each.append(boxes[index])
                scores_each.append(each_clas_score[index])
                if len(boxes_each) == keep_num_postNMS:
                    break

                # Compute ordinates of intersection-over-union(IOU)
                x1 = torch.maximum(start_x_torched[index], start_x_torched[order_torched[:-1]])
                x2 = torch.minimum(end_x_torched[index], end_x_torched[order_torched[:-1]])
                y1 = torch.maximum(start_y_torched[index], start_y_torched[order_torched[:-1]])
                y2 = torch.minimum(end_y_torched[index], end_y_torched[order_torched[:-1]])

                # Compute areas of intersection-over-union
                w = torch.maximum(torch.tensor([0]), x2 - x1 + 1)
                h = torch.maximum(torch.tensor([0]), y2 - y1 + 1)
                intersection = w * h

                # Compute the ratio between intersection and union
                ratio = intersection / (areas_torched[index] + areas_torched[order_torched[:-1]] - intersection)
                left = torch.where(ratio < IOU_thres)[0]
                order_torched = order_torched[left]
            clas_each = torch.tensor([i]*len(scores_each))
            boxes_each = torch.tensor(boxes_each)
            scores_each = torch.tensor(scores_each)
            
            clas_all.append(clas_each)
            boxes_all.append(boxes_each)
            scores_all.append(scores_each)

        final_boxes = torch.cat(boxes_all)
        final_scores = torch.cat(scores_all)
        final_clas = torch.cat(clas_all)
        sorted_final_scores, sorted_final_scores_idx = torch.sort(final_scores, descending=True)
        sorted_final_boxes = final_boxes[sorted_final_scores_idx]
        sorted_final_clas = final_clas[sorted_final_scores_idx]

        return sorted_final_clas, sorted_final_scores, sorted_final_boxes
        # return clas_all, scores_all, boxes_all




    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
    #################################################
    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
        labels_all = labels.flatten()
        # print("Labels : ", labels_all)

        neg_mask = labels_all < 0.
        pos_mask = labels_all >= 0.
        labels_all = torch.where(labels_all < 0., 0., labels_all)

        num_neg_ind = neg_mask.sum().item()
        num_pos_ind = labels_all.shape[0] - num_neg_ind
        print("Num neg ind : ", num_neg_ind)
        print("Num pos ind : ", num_pos_ind)

        if num_pos_ind > (3*effective_batch/4):
            rand_pos_idx = torch.randperm(num_pos_ind)[:int(3*effective_batch/4)]
            rand_neg_idx = torch.randperm(num_neg_ind)[:int(effective_batch/4)]
            pos_clas_tar = labels_all[pos_mask][rand_pos_idx]
            neg_clas_tar = labels_all[neg_mask][rand_neg_idx]
            pos_clas_pred = class_logits[pos_mask][rand_pos_idx]
            neg_clas_pred = class_logits[neg_mask][rand_neg_idx]
            pos_box_pred = box_preds[pos_mask][rand_pos_idx]
            pos_box_tar = regression_targets[pos_mask][rand_pos_idx]
        else:
            rand_neg_idx = torch.randperm(num_neg_ind)
            pos_clas_tar = labels_all[pos_mask]
            neg_clas_tar = labels_all[neg_mask][rand_neg_idx][:(effective_batch - num_pos_ind)]
            pos_clas_pred = class_logits[pos_mask]
            neg_clas_pred = class_logits[neg_mask][rand_neg_idx][:(effective_batch - num_pos_ind)]
            pos_box_pred = box_preds[pos_mask]
            pos_box_tar = regression_targets[pos_mask]


        clas_preds = torch.vstack((pos_clas_pred,neg_clas_pred)).float()

        clas_tar   = torch.cat((pos_clas_tar,neg_clas_tar)).reshape(-1).long()

        clas_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        
        clas_loss = clas_criterion(clas_preds, clas_tar)

        class_labels_cor = (pos_clas_tar.clone() - 1).int()
        
        print(pos_box_pred)
        print(class_labels_cor)
        box_regression_preds = torch.stack([one_box_regression[4*(class_labels_cor[i]):4*class_labels_cor[i]+4] for i, one_box_regression in enumerate(pos_box_pred)])

        reg_criterion = torch.nn.SmoothL1Loss(reduction = 'mean')
        loss_regr = reg_criterion(box_regression_preds,pos_box_tar) 

        loss = clas_loss + l*loss_regr 

        return loss, clas_loss, loss_regr
        

    def forward(self, feature_vectors):
        """
        Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
        Input:
              feature_vectors: (total_proposals, 256*P*P)
        Outputs:
              class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, 
                            notice if you want to use CrossEntropyLoss you should not pass the output through softmax here)
              box_pred:     (total_proposals,4*C)
        """

        #TODO forward through the Intermediate layer
        X = self.intermediate_layer(feature_vectors)

        #TODO forward through the Classifier Head
        class_logits = self.classifier_head(X)

        if eval==True:
          softmax = torch.nn.Softmax(dim = 1)
          class_logits = softmax(class_logits)

        #TODO forward through the Regressor Head
        box_pred = self.regressor_head(X)

        return class_logits, box_pred

if __name__ == '__main__':
    box_head_obj = BoxHead()
    ############################################
    prop = [torch.tensor([[13., 55.,167., 290.],[28., 75.,257., 490.],[20., 35., 186., 320.],[25., 45.,307., 330.],[26., 44.,181., 350.]]), torch.tensor([[13., 55.,167., 290.],[28., 75.,257., 490.],[20., 35., 186., 320.],[25., 45.,307., 330.],[26., 44.,181., 350.]])]
    # bbox = [torch.tensor([[10., 45.,370., 350.],[28.5, 85.2, 287., 490.]])]
    bbox = [torch.tensor([[10., 45.,370., 350.],[28.5, 85.2, 287., 490.],[48.5, 85.2, 217., 450.]]), torch.tensor([[10., 45.,370., 350.],[28.5, 85.2, 287., 490.],[48.5, 85.2, 217., 450.]])]

    # gt_labels = [torch.tensor([2.,1.]), torch.tensor([2.,1.])]
    gt_labels = [torch.tensor([0.,1.,2.]), torch.tensor([2.,0.,2.])]

    labels1,regressor_target1 = box_head_obj.create_ground_truth(prop,gt_labels,bbox)

    print("First returned labels : ",  labels1)
    print("First returned regressor target : ", regressor_target1)

    raise
    ##############################################

    ##############################################
    # proposals = [torch.tensor([[748.8009, 352.6240, 807.4326, 570.0834],
    #     [341.8564,  40.5251, 705.7468, 776.0772],
    #     [744.9592, 323.4161, 813.0808, 671.0620],
    #     [235.8825,  46.3311, 872.8674, 774.7207],
    #     [457.2717,  99.9193, 792.9594, 800.0000],
    #     [421.6976, 215.8569, 672.7389, 780.2885],
    #     [458.6153, 418.3247, 624.3083, 777.1409],
    #     [346.0834, 229.7976, 603.6425, 714.9478],
    #     [732.1453, 318.1639, 818.8305, 745.1012],
    #     [613.8575, 145.2507, 844.2435, 800.0000],
    #     [732.6243, 329.2776, 797.0455, 607.8660],
    #     [759.0717, 368.8920, 806.8002, 536.4076],
    #     [311.8525,  89.6907, 813.9715, 637.9915],
    #     [163.7263, 266.8454, 879.8873, 681.9946],
    #     [239.2761,  44.4321, 579.2953, 800.0000],
    #     [755.1735, 299.4115, 810.2717, 800.0000],
    #     [404.6874, 160.8955, 650.5888, 644.8977],
    #     [281.9656, 223.8601, 786.6899, 799.0535],
    #     [427.5356, 291.7365, 609.1636, 769.6465],
    #     [456.5587, 528.2834, 480.0902, 712.3279]])]

    # torch.manual_seed(1)
    # fpn_list = [torch.rand(1,256, 200,272), torch.rand(1,256, 100,136), torch.rand(1,256, 50,68),torch.rand(1,256, 25,34),torch.rand(1,256, 13,17)]

    # feature_vectors1 = box_head_obj.MultiScaleRoiAlign(fpn_list,proposals,P=7)


    raise
    ################################################################

    ######################################################
    # prop = [torch.tensor([[13., 55.,167., 290.],[28., 75.,257., 490.],[20., 35., 186., 320.],[25., 45.,307., 330.],[26., 44.,181., 350.]]), torch.tensor([[13., 55.,167., 290.],[28., 75.,257., 490.],[20., 35., 186., 320.],[25., 45.,307., 330.],[26., 44.,181., 350.]])]
    prop = [torch.tensor([[748.8009, 352.6240, 807.4326, 570.0834],
        [341.8564,  40.5251, 705.7468, 776.0772],
        [744.9592, 323.4161, 813.0808, 671.0620],
        [235.8825,  46.3311, 872.8674, 774.7207],
        [457.2717,  99.9193, 792.9594, 800.0000]]), ]
    torch.manual_seed(2)
    class_logits = torch.rand(5,4)
    box_regressors = torch.rand(5,12)

    # prop = [torch.tensor([[13., 55.,167., 290.],[28., 75.,257., 490.],[20., 35., 186., 320.],[25., 45.,307., 330.],[26., 44.,181., 350.]])]
    # torch.manual_seed(2)
    # class_logits = torch.rand(5,4)
    # box_regressors = torch.rand(5,12)

    boxes, scores, labels = box_head_obj.postprocess_detections(class_logits, box_regressors, prop)


    




    ######################################################


    raise

def greedy_nms(self,clases,boxes,scores, IOU_thres=0.5, keep_num_postNMS=100):
        # Input:
        #       clases: (num_preNMS, )
        #       boxes:  (num_preNMS, 4)
        #       scores: (num_preNMS,)
        # Output:
        #       boxes:  (post_NMS_boxes_per_image,4) ([x1,y1,x2,y2] format)
        #       scores: (post_NMS_boxes_per_image)   ( the score for the top class for the regressed box)
        #       labels: (post_NMS_boxes_per_image)  (top category of each regressed box)
        ###########################################################

        scores_all = [[],[],[]]
        boxes_all = [[],[],[]]
        clas_all = [[],[],[]]

        for i in range(3):
            each_label_idx = torch.where(clases == i)[0]
            if len(each_label_idx) == 0:
              continue
            each_clas_boxes = boxes[each_label_idx]
            each_clas_score = scores[each_label_idx]

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
                # boxes_each.append(boxes[index])
                # scores_each.append(each_clas_score[index])
                boxes_all[i].append(boxes[index].detach())
                scores_all[i].append(each_clas_score[index].detach())

                if len(boxes_all[i]) == keep_num_postNMS:
                    break

                # Compute ordinates of intersection-over-union(IOU)
                x1 = torch.maximum(start_x_torched[index], start_x_torched[order_torched[:-1]])
                x2 = torch.minimum(end_x_torched[index], end_x_torched[order_torched[:-1]])
                y1 = torch.maximum(start_y_torched[index], start_y_torched[order_torched[:-1]])
                y2 = torch.minimum(end_y_torched[index], end_y_torched[order_torched[:-1]])

                # Compute areas of intersection-over-union
                w = torch.maximum(torch.tensor([0]), x2 - x1 + 1)
                h = torch.maximum(torch.tensor([0]), y2 - y1 + 1)
                intersection = w * h

                # Compute the ratio between intersection and union
                ratio = intersection / (areas_torched[index] + areas_torched[order_torched[:-1]] - intersection)
                left = torch.where(ratio < IOU_thres)[0]
                order_torched = order_torched[left]
            clas_all[i] = [i]*len(scores_all[i])
            
        fin_scores = torch.cat([torch.tensor(one_score).reshape(-1,1) for one_score in scores_all if len(one_score)!=0],dim=0).reshape(-1,1)
        fin_boxes = torch.cat([torch.stack(one_box) for one_box in boxes_all if len(one_box)!=0]).reshape(-1,4)
        fin_clas = torch.cat([torch.tensor(one_clas) for one_clas in clas_all if len(one_clas)!=0]).reshape(-1,1)
        return fin_clas, fin_scores, fin_boxes



    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=20):
        # This function does the post processing for the results of the Box Head for a batch of images
        # Use the proposals to distinguish the outputs from each image
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        
        class_logits = class_logits.cpu()
        box_regression = box_regression.cpu()
        num_proposals = proposals[0].shape[0]
        boxes = []
        scores = []
        labels = []
        for i, each_proposal in enumerate(proposals):
            each_proposal = each_proposal.cpu()
            one_image_boxes = box_regression[i*num_proposals:(i+1)*num_proposals]          # Shape (num_proposals, 12)
            one_image_logits = class_logits[i*num_proposals:(i+1)*num_proposals]           # Shape (num_proposals, 4)
            one_image_scores, one_image_label = torch.max(one_image_logits, dim=1)
            one_image_label = one_image_label.clone().int() - 1
            non_bg_label_idx = torch.where(one_image_label >= 0)[0].cpu()

          
            if len(non_bg_label_idx) != 0: 
                class_labels = one_image_label[non_bg_label_idx]
                all_class_boxes = one_image_boxes[non_bg_label_idx]
                class_boxes =  torch.stack([all_class_boxes[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])      # Shape(filtered_labels, 4) ([t_x,t_y,t_w,t_h])
                
                decoded_boxes = output_decoding_postprocess(class_boxes, each_proposal[non_bg_label_idx])                          # (x1,y1,x2,y2)
                decoded_boxes = decoded_boxes.cpu()

                valid_boxes_idx = torch.where((decoded_boxes[:,0] >= 0) & (decoded_boxes[:,2] < 1088) & (decoded_boxes[:,1] > 0) & (decoded_boxes[:,3] < 800))

                valid_boxes = decoded_boxes[valid_boxes_idx]
                valid_clases = one_image_label[non_bg_label_idx][valid_boxes_idx]
                valid_scores = one_image_scores[non_bg_label_idx][valid_boxes_idx]
                sorted_scores_pre_nms, sorted_idx = torch.sort(valid_scores, descending=True)
                sorted_clases_pre_nms = valid_clases[sorted_idx]
                sorted_boxes_pre_nms = valid_boxes[sorted_idx]
                
                if len(sorted_clases_pre_nms) > keep_num_preNMS:
                    clases_pre_nms = sorted_clases_pre_nms[:keep_num_preNMS]
                    boxes_pre_nms = sorted_boxes_pre_nms[:keep_num_preNMS]
                    scores_pre_nms = sorted_scores_pre_nms[:keep_num_preNMS]
                else:
                    clases_pre_nms = sorted_clases_pre_nms
                    boxes_pre_nms = sorted_boxes_pre_nms
                    scores_pre_nms = sorted_scores_pre_nms
                clases_post_nms, scores_post_nms, boxes_post_nms = greedy_nms(clases_pre_nms, boxes_pre_nms, scores_pre_nms, IOU_thres=conf_thresh, keep_num_postNMS=keep_num_postNMS)
            boxes.append(boxes_post_nms)
            scores.append(scores_post_nms)
            labels.append(clases_post_nms)

        return boxes, scores, labels