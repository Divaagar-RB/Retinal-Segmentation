import argparse, shutil, cv2, os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.datasets import *
from lib.EvalMetric import cal_metrics, normalization
from model.UNet import UNet

# -------------------- Setup --------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="./Results/", help="Path to save test results")
parser.add_argument("--data_type", type=str, default="Fundus", help="Fundus, OCTA, 2PFM")
parser.add_argument("--data_name", type=str, default="DRIVE", help="DRIVE, STARE, ROSE1, OCTA500")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth file")
parser.add_argument("--input_image", type=str, help="Path to a single image for testing")
opt = parser.parse_args()

# -------------------- Model --------------------
generator = UNet(n_channels=3, n_classes=1)

# -------------------- Test Function --------------------
def Test(best_pretrain, test_save_path, input_image=None):
    cuda = torch.cuda.is_available()
    transforms_ = [transforms.ToTensor()]
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Load model checkpoint
    if not os.path.exists(best_pretrain):
        raise FileNotFoundError(f"Checkpoint file not found: {best_pretrain}")
    generator.load_state_dict(torch.load(best_pretrain, map_location='cuda' if cuda else 'cpu'))
    generator.eval()

    # Prepare directories
    temp_path = os.path.join(test_save_path, "temp")
    eval_path = os.path.join(test_save_path, "Eval")
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    # Determine test files
    if input_image:
        test_files = [input_image]
        test_dir = ""  # full path already given
    else:
        test_path_dir = r"D:\VC\Segmentation\data\test\images"
        test_files = [f for f in os.listdir(test_path_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", "tif", "gif"))]
        test_dir = test_path_dir

    transform = transforms.Compose(transforms_)

    # Process each test image
    for test_file in test_files:
        open_name = test_file if input_image else os.path.join(test_dir, test_file)
        base, ext = os.path.splitext(os.path.basename(test_file))
        save_name = os.path.join(temp_path, base + ".png")

        img = Image.open(open_name).convert("RGB")
        img = np.array(img.resize([512, 512]))
        img = transform(img).unsqueeze(0)
        img = Variable(img.type(Tensor))

        with torch.no_grad():
            pred = generator(img)
        pred = pred.data.squeeze().cpu().numpy()
        pred = normalization(pred)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        res_image = (pred * 255).astype(np.uint8)
        cv2.imwrite(save_name, res_image)

    # Move to Eval folder
    for img_file in test_files:
        base, ext = os.path.splitext(os.path.basename(img_file))
        old = os.path.join(temp_path, base + ".png")
        new = os.path.join(eval_path, base + ".png")
        shutil.move(old, new)

    # Return the path of the processed image
    base, ext = os.path.splitext(os.path.basename(test_files[0]))
    result_file = os.path.join(eval_path, base + ".png")
    return result_file

# -------------------- Main --------------------
if __name__ == "__main__":
    result = Test(best_pretrain=opt.checkpoint, test_save_path=opt.save_path, input_image=opt.input_image)
    print("Processed image saved at:", result)
