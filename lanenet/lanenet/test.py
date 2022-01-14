import time
import os
import sys
import argparse

import torch
from torch.autograd import Variable
import numpy as np
import cv2
from torchvision import transforms

from lanenet.model.model import compute_loss
from lanenet.utils.average_meter import AverageMeter
from lanenet.dataloader.transformers import Rescale

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

VGG_MEAN = [103.939, 116.779, 123.68]

# ['instance_seg_logits', 'binary_seg_pred', 'binary_seg_logits']
#   compose_img(image, pred["binary_seg_pred"], pred["binary_seg_logits"], pred["instance_seg_logits"], instance_label, 0)
def compose_img(image_data, out, binary_label, pix_embedding, instance_label, i):
    val_gt = (image_data.cpu().numpy().transpose(1, 2, 0) + VGG_MEAN).astype(np.uint8)
    val_pred = out.squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_label = binary_label.squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_out = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
    val_out[:, :, 0] = val_pred
    val_out[:, :, 1] = val_label
    val_gt[val_out == 255] = 255
    # epsilon = 1e-5
    # pix_embedding = pix_embedding.data.cpu().numpy()
    # pix_vec = pix_embedding / (np.sum(pix_embedding, axis=0, keepdims=True) + epsilon) * 255
    # pix_vec = np.round(pix_vec).astype(np.uint8).transpose(1, 2, 0)
    # ins_label = instance_label.data.cpu().numpy().transpose(0, 1)
    # ins_label = np.repeat(np.expand_dims(ins_label, -1), 3, -1)
    # val_img = np.concatenate((val_gt, pix_vec, ins_label), axis=0)
    # val_img = np.concatenate((val_gt, pix_vec), axis=0)
    # return val_img
    return val_gt

def test(val_loader, model, epoch):
    model.eval()
    step = 0
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    mean_iou = AverageMeter()
    end = time.time()
    val_img_list = []
    # val_img_md5 = open(os.path.join(im_path, "val_" + str(epoch + 1) + ".txt"), "w")
    for batch_idx, input_data in enumerate(val_loader):
        step += 1
        image_data = Variable(input_data["input_tensor"]).to(DEVICE)
        instance_label = Variable(input_data["instance_label"]).to(DEVICE)
        binary_label = Variable(input_data["binary_label"]).to(DEVICE)

        # output process
        net_output = model(image_data)
        total_loss, binary_loss, instance_loss, out, val_iou = compute_loss(net_output, binary_label, instance_label)
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(val_iou, image_data.size()[0])

        # if step % 100 == 0:
        #    val_img_list.append(
        #        compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, 0))
        #    val_img_md5.write(input_data["img_name"][0] + "\n")
    #        lane_cluster_and_draw(image_data, net_output["binary_seg_pred"], net_output["instance_seg_logits"], input_data["o_size"], input_data["img_name"], json_path)
    batch_time.update(time.time() - end)
    end = time.time()

    print(
        "Epoch {ep} Validation Report | ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
            ep=epoch + 1,
            et=batch_time.val,
            tot=total_losses.avg,
            bin=binary_losses.avg,
            ins=instance_losses.avg,
            iou=mean_iou.avg,
        ))
    sys.stdout.flush()
    val_img = np.concatenate(val_img_list, axis=1)
    # cv2.imwrite(os.path.join(im_path, "val_" + str(epoch + 1) + ".png"), val_img)
    # val_img_md5.close()
    return mean_iou.avg

if __name__ == "__main__":

    # python lanenet/test.py ./checkpoints/4_checkpoint.pth
    parser = argparse.ArgumentParser(description='Lane Detection')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint file. Checkpoint should be on the same path.'
    )
    args = parser.parse_args()
    model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    # print(model)
    transform=transforms.Compose([Rescale((512, 256))])
    image = cv2.imread('D:/College/SEM-VII/PBS/vehicle_automation/lanenet/data/tusimple_test_image/0.jpg', cv2.IMREAD_COLOR)
    img = transform(image)
    img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
    img = torch.Tensor([img])
    img = Variable(img).type(torch.FloatTensor)
    # ['instance_seg_logits', 'binary_seg_pred', 'binary_seg_logits']
    pred = model(img)
    # new_img = compose_img(torch.Tensor(img), pred["binary_seg_pred"], pred["binary_seg_logits"], pred["instance_seg_logits"], None, 0)

    cv2.imwrite("train_pred.png", pred["binary_seg_pred"].squeeze(0).cpu().numpy().transpose(0,1)*255)
    # cv2.imshow("s", new_img)

    # cv2.waitKey(0)
