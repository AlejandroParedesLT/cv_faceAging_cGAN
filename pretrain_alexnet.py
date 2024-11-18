import torch
import torchvision
import torch.utils.data as data
import os
from os.path import join
import argparse
import logging
from tqdm import tqdm
# User imports
from data_generator.DataLoader_Pretrain_Alexnet import CACD
#from model.faceAlexnet import AgeClassify
from model.pretrainedAgeAlexnet import AgeClassify
from utils.io import check_dir, Img_to_zero_center
from PIL import Image
from datetime import datetime

# Step 1: Define argument parser
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

parser = argparse.ArgumentParser(description='Pretrain age classifier')
# Optimizer
parser.add_argument('--learning_rate', '--lr', type=float, help='Learning rate', default=1e-4)
parser.add_argument('--batch_size', '--bs', type=int, help='Batch size', default=512)
parser.add_argument('--max_epoches', type=int, help='Number of epochs to run', default=10)
parser.add_argument('--val_interval', type=int, help='Number of steps to validate', default=10)
parser.add_argument('--save_interval', type=int, help='Number of batches to save model', default=10)

# Model
# Data and IO
parser.add_argument('--cuda_device', type=str, help='Which device to use', default='0')
parser.add_argument('--checkpoint', type=str, help='Logs and checkpoints directory', default='./checkpoint/pretrain_alexnet')
parser.add_argument('--saved_model_folder', type=str,
                    help='Path to the folder that stores the parameter files',
                    default='./checkpoint/pretrain_alexnet/saved_parameters/%s/'%(TIMESTAMP))
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

check_dir(args.checkpoint)
check_dir(args.saved_model_folder)

# Step 2: Define logging output
logger = logging.getLogger("Age Classifier")
file_handler = logging.FileHandler(join(args.checkpoint, 'log.txt'), "w")
stdout_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.setLevel(logging.INFO)

# Step 3: Update the CACD Dataset Class
from PIL import ImageOps

class CACDPreprocessed(CACD):
    def __init__(self, split='train', transforms=None, label_transforms=None):
        super().__init__(split, transforms, label_transforms)  # Call the parent constructor

    def __getitem__(self, idx):
        # Access images_labels from the parent class (CACD)
        img_path, label = self.images_labels[idx]
        img = Image.open(img_path)

        # Ensure the image is in RGB format (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label



def main():
    logger.info("Start to train:\n Arguments: %s" % str(args))
    # Step 4: Define transform
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((227, 227)),
        torchvision.transforms.ToTensor(),
        Img_to_zero_center()
    ])
    # Step 5: Define train/test dataloader with the updated dataset class
    train_dataset = CACDPreprocessed("train", transforms, None)
    test_dataset = CACDPreprocessed("test", transforms, None)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    # Step 6: Define model and optimizer
    model = AgeClassify()
    optim = model.optim

    for epoch in range(args.max_epoches):
        for train_idx, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()

            # Train step
            optim.zero_grad()
            model.train(img, label)
            loss = model.loss
            loss.backward()
            optim.step()
            format_str = ('Step %d/%d, cls_loss = %.3f')
            logger.info(format_str % (train_idx, len(train_loader), loss))

            # Save the parameters at the end of each save interval
            if train_idx * args.batch_size % args.save_interval == 0:
                model.save_model(dir=args.saved_model_folder, 
                                 filename='epoch_%d_iter_%d.pth' % (epoch, train_idx))
                logger.info('Checkpoint has been created!')

            # Validation step
            if train_idx % args.val_interval == 0:
                train_correct = 0
                train_total = 0
                with torch.no_grad():
                    for val_img, val_label in tqdm(test_loader):
                        val_img = val_img.cuda()
                        val_label = val_label.cuda()
                        output = model.val(val_img)
                        train_correct += (output == val_label).sum()
                        train_total += val_img.size()[0]

                logger.info('Validation has been finished!')
                format_str = ('val_acc = %.3f')
                logger.info(format_str % (train_correct.cpu().numpy() / train_total))

if __name__ == '__main__':
    main()
