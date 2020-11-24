import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from makegri import make_grid
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import joint_transforms
from config import sbu_training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model import BDRAR

# cudnn.benchmark = True

torch.cuda.set_device(0)

ckpt_path = 'ckpt\\'
exp_name = 'BDRAR'
writer = SummaryWriter()

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'iter_num': 70000,
    'train_batch_size': 16,
    'last_iter': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 256
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(sbu_training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)  # 改

bce_logit = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = BDRAR().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '17000.pth')))
        # optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        # optimizer.param_groups[0]['lr'] = 2 * args['lr']
        # optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    train(net, optimizer)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            predict0, predict1, predict2, predict3, predict4, predict5 = net(inputs)

            loss0 = bce_logit(predict0, labels)
            loss1 = bce_logit(predict1, labels)
            loss2 = bce_logit(predict2, labels)
            loss3 = bce_logit(predict3, labels)
            loss4 = bce_logit(predict4, labels)
            loss5 = bce_logit(predict5, labels)

            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()

            optimizer.step()

            if curr_iter % 50 == 0:
                p0 = make_grid(predict0, normalize=True)
                p1 = make_grid(predict1, normalize=True)
                p2 = make_grid(predict2, normalize=True)
                labe = make_grid(labels, normalize=True)
                writer.add_image('p00', p0, curr_iter)
                writer.add_image('p11', p1, curr_iter)
                writer.add_image('p22', p2, curr_iter)
                writer.add_image('labe', labe, curr_iter)
                writer.add_scalar('loss', loss, curr_iter)

            curr_iter += 1
            log = '[iter %d], [train loss0 %.5f], [loss1 %.5f], [loss2 %.5f],[loss3 %.5f], [loss4 %.5f], [lr %.13f]' % \
                  (curr_iter, loss0, loss1, loss2, loss3, loss4, optimizer.param_groups[1]['lr'])
            print(log)
            if curr_iter % 1000 == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))


if __name__ == '__main__':
    main()
