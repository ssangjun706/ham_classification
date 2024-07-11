image_dir = "dataset/imgs"
label_path = "dataset/label.csv"

batch_size = 256
epochs = 25
lr = 1e-3

patch_size = 16
h_dim = 768
mlp_dim = 3072
image_size = 432
num_classes = 7

num_workers = 32
use_wandb = True
use_checkpoint = False

model = "vit"

if model == "vit":
    checkpoint = "checkpoint/checkpoint_model_vit.pt"
else:
    checkpoint = "checkpoint/checkpoint_model_resnet.pt"
