import torch.nn as nn
import torch
import numpy as np
import sys
from utils import utils
from torch.autograd import Variable

class YOLOLayer(nn.Module):
    def __init__(self, anchors):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
        self.ce_loss = nn.CrossEntropyLoss() 
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def forward(self, prediction, img_size, num_classes, CUDA, targets=None):
        anchors = self.anchors
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride =  img_size // grid_size
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)

        prediction = prediction.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

        scaled_anchors = torch.FloatTensor([(a[0]/stride, a[1]/stride) for a in anchors])

        #Sigmoid the  centre_X, centre_Y. and object confidencce
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if CUDA:
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
            ByteTensor = torch.cuda.ByteTensor
            scaled_anchors = scaled_anchors.type(FloatTensor)
        
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)
          
        anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = utils.buildTargets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                targets=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=num_anchors,
                num_classes=num_classes,
                grid_size=grid_size,
                ignore_thres=0.5,
                img_size=img_size,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.mse_loss(x[mask], tx[mask])
            loss_y = self.lambda_coord * self.mse_loss(y[mask], ty[mask])
            loss_w = self.lambda_coord * self.mse_loss(w[mask], tw[mask])
            loss_h = self.lambda_coord * self.mse_loss(h[mask], th[mask])

            loss_conf = self.lambda_noobj * self.mse_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.mse_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = (1 / batch_size) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision
            )

        else:
            prediction = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,\
                        pred_conf.view(batch_size, -1, 1),\
                        pred_cls.view(batch_size, -1, num_classes)),-1)
            prediction = prediction.view(batch_size, num_anchors, grid_size, grid_size, bbox_attrs).permute(0, 2,3,1,4)\
                        .contiguous().view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
            return prediction