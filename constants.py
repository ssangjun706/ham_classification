image_dir = "../../data/ham10000/imgs"
label_path = "../../data/ham10000/label.csv"

batch_size = 256
epochs = 10
lr = 1e-3

patch_size = 16
h_dim = 768
mlp_dim = 3072
image_size = 450
num_classes = 7

num_workers = 32
use_wandb = False
use_checkpoint = False

model = "vit"

if model == "vit":
    checkpoint = "checkpoint/checkpoint_model_vit.pt"
else:
    checkpoint = "checkpoint/checkpoint_model_resnet.pt"
