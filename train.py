import argparse, shutil, cv2, time, datetime, sys, os, gc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lib.datasets import *
from lib.Loss import initialize_weights, Connect_Loss
from lib.EvalMetric import cal_metrics_train, normalization
from model.UNet import UNet
# from model.csnet import CSNet
# from model.vision_transformer import SwinUnet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--save_path", type=str, default="./Results/", help="path of results after test")
parser.add_argument("--data_type", type=str, default="Fundus", help="Fundus, OCTA, 2PFM")
parser.add_argument("--data_name", type=str, default="DRIVE", help="DRIVE, STARE, ROSE1, OCTA500")
parser.add_argument('--mask_type', type=str, default="MaskVSC", help='MaskVSC, or None')
opt = parser.parse_args()

train_path = r"D:\VC\MaskVSC\data\training"
test_path  = r"D:\VC\MaskVSC\data\test\test"


# ---Eval---
# ---Eval---
def Eval(test_path, test_save_path):
    import numpy as np
    from PIL import Image

    transforms_ = [transforms.ToTensor()]
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    test_dir = os.path.join(test_path, "images")
    test_files = os.listdir(test_dir)
    transform = transforms.Compose(transforms_)

    Eval_path = os.path.join(test_save_path, "Eval")
    temp_path = os.path.join(test_save_path, "temp")

    # Clean and create directories
    for path in [Eval_path, temp_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    for test_file in test_files:
        open_name = os.path.join(test_dir, test_file)
        save_name = os.path.join(temp_path, test_file)

        # Fundus
        img = Image.open(open_name)
        img = np.array(img.resize([512, 512]))

        # OCTA & 2PFM (uncomment if needed)
        # img = cv2.imread(open_name, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (384,384)) # OCTA500

        img = transform(img).unsqueeze(0)
        img = Variable(img.type(Tensor))

        # Generate prediction
        pred = generator(img)
        pred = pred.data.squeeze().cpu().numpy()
        pred = normalization(np.array(pred))
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        res_image = (pred * 255).astype(np.uint8)
        cv2.imwrite(save_name, res_image)

    # Move results to Eval folder
    for img_file in test_files:
        old = os.path.join(temp_path, img_file)
        new = os.path.join(Eval_path, img_file)
        shutil.move(old, new)

    # Calculate evaluation metrics
    GT_path = os.path.join(test_path, "mask")
    Dice = cal_metrics_train(opt.data_type, Eval_path, GT_path)

    return Dice


# ---Training---
def train(generator, criterion_BCE, criterion_Con, optimizer):
    ...

    transforms_ = [transforms.ToTensor(),]
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    prev_time = time.time()
    best_dice = 0.0

    for epoch in range(opt.n_epochs):
        # ---Masking Ratio---
        epoch_ratio = epoch/opt.n_epochs
        if 0 <= epoch_ratio < 0.1 or 0.9 < epoch_ratio <= 1:
            curr_mask_ratio = 0
        elif 0.1 <= epoch_ratio <= 0.5:
            curr_mask_ratio = 0.4 * (epoch_ratio - 0.1) / 0.4
        elif 0.5 < epoch_ratio <= 0.9:
            curr_mask_ratio = 0.4 * (0.9 - epoch_ratio) / 0.4
        else:
            raise Exception("Invalid epoch_ratio!", epoch_ratio)

        # ---Dataloader---
        if opt.data_type == "Fundus":
            train_dataloader = DataLoader(
                ImageDataset_Fundus(train_path, transforms_=transforms_,
                                  mask_type=opt.mask_type, mask_ratio=curr_mask_ratio),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        elif opt.data_type == "OCTA":
            train_dataloader = DataLoader(
                ImageDataset_OCTA(train_path, transforms_=transforms_,
                                  mask_type=opt.mask_type, mask_ratio=curr_mask_ratio),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        elif opt.data_type == "2PFM":
            train_dataloader = DataLoader(
                ImageDataset_2PFM(train_path, transforms_=transforms_,
                                  mask_type=opt.mask_type, mask_ratio=curr_mask_ratio),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,)
        else:
            raise Exception("Invalid data type!", opt.data_type)

        # ---Training one epoch---
        for i, batch in enumerate(train_dataloader):
            img = Variable(batch["image"].type(Tensor))
            lab = Variable(batch["label"].type(Tensor))
            
            pred = generator(img)
            
            # ---Loss Function---
            loss_BCE = criterion_BCE(pred, lab)
            loss_Con = criterion_Con(pred, lab)

            loss = loss_BCE + loss_Con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---Log Progress---
            batches_done = epoch * len(train_dataloader) + i
            batches_left = opt.n_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s"
                % (epoch, opt.n_epochs, i, len(train_dataloader), loss.item(), time_left))

                # ---best model---
        # --- Evaluate after each epoch ---
        Dice = Eval(test_path, opt.save_path)

        # Print raw Dice for debugging
        print(f"\n[Epoch {epoch}] Raw Dice from Eval: {Dice}")

        # Normalize Dice to [0,1] for logging purposes
        Dice_clamped = max(0.0, min(Dice, 1.0))

        # --- Ensure saved_models folder exists ---
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)

        # --- Save model every epoch ---
        save_path = os.path.join(save_dir, f"Epo{epoch}_Dice{Dice_clamped:.4f}.pth")
        torch.save(generator.state_dict(), save_path)
        print(f"============ Saved model at epoch {epoch}: {save_path} ============")

        # --- Update best Dice if improved ---
        if Dice_clamped > best_dice:
            best_dice = Dice_clamped
            print(f"============ New best Dice! ({best_dice:.4f}) ============")

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

       


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    global generator, criterion_BCE, criterion_Con, optimizer
        # ---Model, Loss, Optimizer---
    generator = UNet(n_channels=3, n_classes=1)
    initialize_weights(generator)

    criterion_BCE = nn.BCELoss()
    criterion_Con = Connect_Loss()

    optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator = generator.cuda()
        criterion_BCE.cuda()
        criterion_Con.cuda()


    train(generator, criterion_BCE, criterion_Con, optimizer)

