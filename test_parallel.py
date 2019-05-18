# Compatability Imports
from __future__ import print_function

import os

DEVICE_IDS = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in DEVICE_IDS])

# ability to perform multiprocessing
import multiprocessing
from joblib import Parallel, delayed
# use threading instead
#from joblib.pool import has_shareable_memory

NUM_CORES = multiprocessing.cpu_count()
print(
    "Preprocessing will run on {} CPU cores on your machine.".format(NUM_CORES)
)

from os.path import join
from data import readSEGY, get_slice
from texture_net import TextureNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
from torch.autograd import Variable
import tb_logger
from data import writeSEGY

# graphical progress bar for notebooks
from tqdm import tqdm

# Parameters
DATASET_NAME = "F3"
IM_SIZE = 65
N_CLASSES = 2
RESOLUTION = 1
SLICE = "inline"  # Inline, crossline, timeSLICE or full
SLICE_NUM = 339
#BATCH_SIZE = 2**12 # 4096 memory limit
BATCH_SIZE = 4

# use distributed scoring
if RESOLUTION != 1:
    raise Exception("Currently we only support pixel-level scoring")

# Read 3D cube
data, data_info = readSEGY(join(DATASET_NAME, "data.segy"))

# Load trained model (run train.py to create trained
network = TextureNet(n_classes=N_CLASSES)
network.load_state_dict(torch.load(join(DATASET_NAME, "saved_model.pt")))


class ModelWrapper(nn.Module):
    """
  Wrap TextureNet for DataParallel to invoke classify method
  """

    def __init__(self, texture_model):
        super(ModelWrapper, self).__init__()
        self.texture_model = texture_model

    def forward(self, input):
        return self.texture_model.classify(input)


if torch.cuda.is_available():
    # yup, apparently for data parallel models this has cuda:0... oh well
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    raise Exception("No GPU detected for parallel scoring!")

network.eval()

model = ModelWrapper(network)
model.eval()

# We can set the interpretation RESOLUTION to save time.f
# The interpretation is then conducted over every n-th sample and
# then resized to the full size of the input data
print("RESOLUTION {}".format(RESOLUTION))

##########################################################################

# Log to tensorboard
logger = tb_logger.TBLogger("log", "Test")
logger.log_images(
    SLICE + "_" + str(SLICE_NUM),
    get_slice(data, data_info, SLICE, SLICE_NUM),
    cm="gray",
)

# classified_cube = interpret(network.classify, data, data_info, 'full', None, IM_SIZE, RESOLUTION, use_gpu=use_gpu)
# model = nn.DataParallel(network.classify)

# Get half window size

window = IM_SIZE // 2
nx, ny, nz = data.shape

# generate full list of coordinates
# memory footprint of this isn't large yet, so not need to wrap as a generator
x_list = range(window, nx - window + 1)
y_list = range(window, ny - window + 1)
z_list = range(window, nz - window + 1)

print("-- generating coord list --")
# TODO: is there any way to use a generator with pyTorch data loader
coord_list = list(itertools.product(x_list, y_list, z_list))

class MyDataset(Dataset):
    def __init__(self, data, window, coord_list):

        # main array
        self.data = data
        self.coord_list = coord_list
        self.window = window
        self.len = len(coord_list)

    def __getitem__(self, index):

        pixel = self.coord_list[index]
        x, y, z = pixel
        small_cube = self.data[
            x - self.window : x + self.window,
            y - self.window : y + self.window,
            z - self.window : z + self.window,
        ]
        # return Variable(torch.FloatTensor(small_cube[np.newaxis, :, :, :]))
        # return torch.Tensor(voxel).float()
        return torch.FloatTensor(small_cube[np.newaxis, :, :, :]), index

    def __len__(self):
        return self.len

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
my_loader = DataLoader(
    dataset=MyDataset(data, window, coord_list), batch_size=BATCH_SIZE, shuffle=False
)

# unroll full cube
indices = []
predictions = []

from time import time

print ("-- scoring on GPU --")
loop = time()
# Loop through center pixels in output cube
for (chunk, index) in tqdm(my_loader):

    #print("loop wait time", time() - loop)
    overall = time()
    input = chunk.to(device)
    start = time()
    output = model(input)
    #print("scoring", time() - start)
    # save and deal with it later on CPU
    start = time()
    indices += index.tolist()
    #print("index_time", time() - start)
    start = time()
    predictions += output.tolist()
    #print("predictions_time", time() - start)


    #print ("ETA estimate", (time()-overall)*BATCH_SIZE/3600)
    loop = time()
    #coords = [list(np.array(pixel[j])) for j in range(3)]
    #xyz = list(zip(*coords))
    #for i in range(BATCH_SIZE):
    #    pred_class = pred[i]
    #    x, y, z = xyz[i]
    #    classified_cube[x,y,z] = pred_class

print ("-- aggregating results --")

classified_cube = np.zeros(data.shape)

def worker(classified_cube, ind):
    x, y, z = coord_list[ind]
    pred_class = predictions[ind][0][0][0][0]
    classified_cube[x, y, z] = pred_class

# process masks first because we COULD subset data on mask operations
_ = Parallel(n_jobs=NUM_CORES, backend = "threading")(
    delayed(worker)(classified_cube, ind)
    for ind in tqdm(indices)
)

print("-- writing segy --")
in_file = join(DATASET_NAME, "data.segy".format(RESOLUTION))
out_file = join(DATASET_NAME, "salt_{}.segy".format(RESOLUTION))
writeSEGY(out_file, in_file, classified_cube)

print("-- logging prediction --")
# log prediction to tensorboard
logger = tb_logger.TBLogger("log", "Test_scored")
logger.log_images(
    SLICE + "_" + str(SLICE_NUM),
    get_slice(classified_cube, data_info, SLICE, SLICE_NUM),
    cm="gray",
)
