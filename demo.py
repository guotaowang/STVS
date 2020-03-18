import numpy as np
import torch
import os, argparse
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from scipy import misc
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
import joint_transforms
from data import test_dataset
import torch
from dataset import ImageFolder
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from model import STFM

torch.cuda.set_device(0)


ckpt_path = 'model\\'
exp_name = ''
args = {
    'snapshot': 'STFA',
    'scale': 256
}

joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.Resize(args['scale']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

to_test = {'sbu': sbu_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = STFM().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))

    net.eval()
    mpath = 'F:\\DAVSOD-master\\Datasets\\STFA\\'
    with torch.no_grad():
        # for target in sorted(os.listdir(mpath)):
            target = 'DAVIS'
            save_path = 'results\\STFA\\STFA_' + target
            image_root = mpath + target + '\\'
            test_set = ImageFolder(image_root, joint_transform, img_transform, target_transform)
            test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
            for _, (image, name, img_name, width, height) in enumerate(test_loader):
                image = image.cuda()
                image = torch.squeeze(image)
                begin = time.time()
                result = net(image)
                print('{:.5f}'.format(time.time()-begin))
                for t in range(result.size(0)):
                    result = F.upsample(result, size=(height[0], width[0]), mode='bilinear', align_corners=False)
                    res = result[t].data.cpu().numpy().squeeze()
                    # res = np.round(res*255)
                    # res = res.astype(np.uint8)
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    save_name = save_path+'\\' + name[0] + '\\'
                    if not os.path.exists(save_name):
                        os.makedirs(save_name)
                    # print(img_name[t][0][:-4])
                    misc.imsave(save_name + img_name[t][0][:-4]+'.png', res)


if __name__ == '__main__':
    main()
