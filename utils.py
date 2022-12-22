import numpy as np
import torch
import torchvision
from sklearn import metrics
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

def IOU(boxA, boxB):
    # This function compute the IOU between two set of boxes 
    boxA = boxA.to(device)
    boxB = boxB.to(device)
    iou = torchvision.ops.box_iou(boxA, boxB)
    return iou


def output_flattening(out_r, out_c, anchors):
    # This function flattens the output of the network and the corresponding anchors
    # in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
    # the FPN levels from all the images into 2D matrices
    # Each row correspond of the 2D matrices corresponds to a specific grid cell
    # Input:
    #       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
    #       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
    #       anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    # Output:
    #       flatten_regr: (total_number_of_anchors*bz,4)
    #       flatten_clas: (total_number_of_anchors*bz)
    #       flatten_anchors: (total_number_of_anchors*bz,4)
    flatten_regr_all = []
    flatten_clas_all = []
    flatten_anchors_all = []
    for level_idx in range(5):
        bz = out_r[level_idx].shape[0]
        flatten_regr = out_r[level_idx].reshape(bz,3,4,out_r[level_idx].shape[-2],out_r[level_idx].shape[-1]).permute(0,1,3,4,2).reshape(-1,4)
        flatten_clas = out_c[level_idx].reshape(-1)
        flatten_anchors = anchors[level_idx].reshape(-1,4).repeat(1,bz)
        flatten_regr_all.append(flatten_regr)
        flatten_clas_all.append(flatten_clas)
        flatten_anchors_all.append(flatten_anchors)
    
    flatten_regr_all = torch.cat(flatten_regr_all)
    flatten_clas = torch.cat(flatten_clas_all)
    flatten_anchors = torch.cat(flatten_anchors_all)

    return flatten_regr_all, flatten_clas, flatten_anchors


def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    # This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
    # into box coordinates where it returns the upper left and lower right corner of the bbox
    # Input:
    #       flatten_out: (total_number_of_anchors*bz,4)
    #       flatten_anchors: (total_number_of_anchors*bz,4)
    # Output:
    #       box: (total_number_of_anchors*bz,4)
    conv_box = torch.zeros_like(flatten_anchors)
    conv_box[:,3] = torch.exp(flatten_out[:,3]) * flatten_anchors[:,3]
    conv_box[:,2] = torch.exp(flatten_out[:,2]) * flatten_anchors[:,2]
    conv_box[:,1] = (flatten_out[:,1] * flatten_anchors[:,3]) + flatten_anchors[:,1]
    conv_box[:,0] = (flatten_out[:,0] * flatten_anchors[:,2]) + flatten_anchors[:,0]

    box = conv_box_to_corners(conv_box)
    return box




# This function converts x,y,w,h to x1,y1,x2,y2
def conv_box_to_corners(box):
    fin_box = torch.zeros_like(box)
    fin_box[:,0] = box[:,0] - (box[:,2]/2)
    fin_box[:,1] = box[:,1] - (box[:,3]/2)
    fin_box[:,2] = box[:,0] + (box[:,2]/2)
    fin_box[:,3] = box[:,1] + (box[:,3]/2)
    return fin_box

# This function converts x1,y1,x2,y2 to x,y,w,h 
def conv_box_to_xywh(box):
    fin_box = torch.zeros_like(box)
    fin_box[:,0] = (box[:,0] + box[:,2]) / 2
    fin_box[:,1] = (box[:,1] + box[:,3]) / 2
    fin_box[:,2] = (box[:,2] - box[:,0])
    fin_box[:,3] = (box[:,3] - box[:,1])
    return fin_box

# This function computes the IOU between two set of boxes
def IOU_gt(boxA, boxB):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes

    boxA_conv = conv_box_to_corners(boxA)
    iou_torched = IOU(boxA_conv, boxB) 
    ##################################
    return iou_torched

def Resnet50Backbone(checkpoint_file=None, device="cpu", eval=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

    if eval == True:
        model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    resnet50_fpn = model.backbone

    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)

        resnet50_fpn.load_state_dict(checkpoint['backbone'])

    return resnet50_fpn

# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding_postprocess(flatten_out,flatten_anchors, device=device):
    #######################################
    # TODO decode the output
    flatten_anchors = conv_box_to_xywh(flatten_anchors)
    conv_box = torch.zeros_like(flatten_anchors).to(device)

    
    conv_box[:,3] = torch.exp(flatten_out[:,3]) * flatten_anchors[:,3]
    conv_box[:,2] = torch.exp(flatten_out[:,2]) * flatten_anchors[:,2]
    conv_box[:,1] = (flatten_out[:,1] * flatten_anchors[:,2]) + flatten_anchors[:,1]
    conv_box[:,0] = (flatten_out[:,0] * flatten_anchors[:,3]) + flatten_anchors[:,0]

    box = conv_box_to_corners(conv_box)

    #######################################
    return box


def pretrained_models_680(checkpoint_file,eval=True):
    import torchvision
    model_fpn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    if(eval):
        model_fpn.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_fpn.to(device)

    backbone = model_fpn.backbone
    rpn = model_fpn.rpn

    if(eval):
        backbone.eval()
        rpn.eval()

    rpn.nms_thresh=0.6
    checkpoint = torch.load(checkpoint_file, device)

    backbone.load_state_dict(checkpoint['backbone'])
    rpn.load_state_dict(checkpoint['rpn'])

    return backbone, rpn


def visualizer_top_proposals(image, boxes, labels):
    labels = np.array(labels)
    boxes = boxes.detach().cpu().numpy()

    image = image.detach().cpu().numpy()
    image = np.clip(image, 0., 255.)

    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot()

    ax.imshow(image)
    for i in range(len(boxes)):

      if labels[i] == 0 :
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='blue')
        ax.add_patch(rect)
      elif labels[i] == 1:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='green')
        ax.add_patch(rect)
      elif labels[i] == 2:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='red')
        ax.add_patch(rect)

def visualize_raw_processor(img, mask,label, alpha=0.5):
    processed_mask = mask.clone().detach().squeeze().bool()
    img = img.clone().detach()[:, :, 11:-11]

    inv_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[ 0., 0., 0. ], std=[1/0.229, 1/0.224, 1/0.255]), torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    pad = torch.nn.ZeroPad2d((11,11,0,0))
    processed_img = pad(inv_transform(img))

    processed_img = processed_img.detach().cpu().numpy()

    processed_img = np.clip(processed_img, 0, 1)
    processed_img = torch.from_numpy((processed_img * 255.).astype(np.uint8))

    img_to_draw = processed_img.detach().clone()

    if processed_mask.ndim == 2:
        processed_mask = processed_mask[None, :, :]
    for mask, lab in zip(processed_mask, label):
        if lab.item() == 0 :            # vehicle
            colored = 'blue'
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
        if lab.item() == 1 :            # person
            colored = 'green'
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
        if lab.item() == 2 :            # animal
            colored = 'red'
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
 
    out = (processed_img * (1 - alpha) + img_to_draw * alpha).to(torch.uint8)
    # out = draw_bounding_boxes(out, bbox, colors='red', width=2)
    final_img = out.numpy().transpose(1,2,0)
    plt.figure()
    plt.imshow(final_img)
    return final_img