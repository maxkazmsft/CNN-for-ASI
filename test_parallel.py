# Compatability Imports
from __future__ import print_function

import os
import pickle
import json
from copy import deepcopy

N_GPU = 2
DEVICE_IDS = list(range(N_GPU))
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in DEVICE_IDS])

import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager

if torch.cuda.is_available():
    device_str = os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device("cuda:"+device_str)
else:
    raise Exception("No GPU detected for parallel scoring!")

# ability to perform multiprocessing
import multiprocessing
from joblib import Parallel, delayed

# use threading instead
# from joblib.pool import has_shareable_memory

from os.path import join
from data import readSEGY, get_slice
from texture_net import TextureNet
import itertools
import numpy as np
import tb_logger
from data import writeSEGY

# graphical progress bar for notebooks
from tqdm import tqdm

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class ModelWrapper(nn.Module):
    """
    Wrap TextureNet for DataParallel to invoke classify method
    """

    def __init__(self, texture_model):
        super(ModelWrapper, self).__init__()
        self.texture_model = texture_model

    def forward(self, input):
        return self.texture_model.classify(input)

class MyDataset(Dataset):
    def __init__(self, data, window, coord_list):

        # main array
        self.data = data
        self.coord_list = coord_list
        self.window = window
        self.len = len(coord_list)

    def __getitem__(self, index):

        # TODO: can we specify a pixel mathematically by index?
        pixel = self.coord_list[index]
        x, y, z = pixel
        # TODO: current bottleneck - can we slice out voxels any faster
        small_cube = self.data[
            x - self.window : x + self.window + 1,
            y - self.window : y + self.window + 1,
            z - self.window : z + self.window + 1,
        ]

        return small_cube[np.newaxis, :, :, :], index

    def __len__(self):
        return self.len

def i_chunk(lst, n, i):
    """Return i-th chunk after splitting the list into n equally sized chunks """
    assert(i<n)
    assert(i>=0)
    avg = len(lst) / float(n)
    last = 0.
    counter = 0
    while last < len(lst):
        chunk = lst[int(last):int(last+avg)]
        if counter == i:
            return chunk
        last += avg
        counter += 1
    return chunk


#def main_worker(gpu, ngpus_per_node, result_queue, args):
def main_worker(gpu, ngpus_per_node, args):

    print ("I got GPU", gpu)
    args.gpu = gpu%ngpus_per_node

    args.rank = gpu
    print("setting rank", args.rank, "world size", args.world_size, args.dist_backend, args.dist_url)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:"+str(args.gpu))

    # Load trained model (run train.py to create trained
    network = TextureNet(n_classes=args.n_classes)
    #import time
    #time.sleep(3600)
    model_state_dict = torch.load(join(args.dataset_name, "saved_model.pt"), map_location=device)
    network.load_state_dict(model_state_dict)
    network.eval()
    network.cuda(args.gpu)

    model = ModelWrapper(network)
    model.eval()
    model.cuda(args.gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = max(int(args.batch_size / ngpus_per_node), 1)
    # number of data loading workers
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    print(type(args.gpu), args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.gpu])

    cudnn.benchmark = True

    # device = torch.device("cuda:" + str(args.gpu))

    # Read 3D cube
    data, data_info = readSEGY(join(args.dataset_name, "data.segy"))

    # Get half window size
    window = args.im_size // 2
    nx, ny, nz = data.shape

    # generate full list of coordinates
    # memory footprint of this isn't large yet, so not need to wrap as a generator
    x_list = range(window, nx - window)
    y_list = range(window, ny - window)
    z_list = range(window, nz - window)

    print("-- generating coord list --")
    # TODO: is there any way to use a generator with pyTorch data loader?
    coord_list = list(itertools.product(x_list, y_list, z_list))

    print("take a subset of coord_list by chunk")
    coord_list = list(np.array_split(np.array(coord_list), args.world_size)[args.rank])
    coord_list = [tuple(x) for x in coord_list]

    # prepare the data
    print ("setup dataset")
    # TODO: RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
    data_torch = torch.cuda.FloatTensor(data).cuda(args.gpu, non_blocking=True)
    dataset = MyDataset(data_torch, window, coord_list)
    # coord_list = dataset.get_coord_list()
    # take a subset of coord_list by chunk
    # coord_list = list(np.array_split(np.array(coord_list), args.world_size)[args.rank])
    # coord_list = [tuple(x) for x in coord_list]
    # dataset.set_coord_list(coord_list)

    # datasampler = DistributedSampler(dataset)
    # just set some default epoch
    #datasampler.set_epoch(1)

    #progress = ProgressMeter(
    #    len(dataset),
    #    [],
    #    prefix="step: ")

    print ("setting up loader")
    my_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=None
        #sampler=datasampler
    )

    print ("running loop")

    #indices = []
    predictions = []

    # Loop through center pixels in output cube
    #for (chunk, index) in tqdm(my_loader):
    with torch.no_grad():
        print ("no grad")
        for (chunk, index) in tqdm(my_loader):
            # print('i', index)
            input = chunk.cuda(args.gpu, non_blocking=True)
            output = model(input)
            # save and deal with it later on CPU
            #indices += index.tolist()
            predictions += output.tolist()
            break
            #if i % args.print_freq == 0:
            #    progress.display(i)

    print("here")
    #result_queue.append([deepcopy(coord_list), deepcopy(predictions)])
    # result_queue.append([coord_list, predictions])
    with open("results_{}.pkl".format(args.rank), "w") as f:
        json.dump({"coords": [(int(x[0]), int(x[1]), int(x[2])) for x in coord_list], "preds": [int(x[0][0][0][0]) for x in predictions]}, f)
    #with open("result_predictions_{}.pkl".format(args.rank), "wb") as f:
    #    print ("dumping predictions pickle file")
    #    pickle.dump(predictions, f)
    print('done')

class Arguments: pass

def main():

    # TODO: wrap into parameters
    # Parameters
    NUM_CORES = multiprocessing.cpu_count()
    print("Post-processing will run on {} CPU cores on your machine.".format(NUM_CORES))
    DATASET_NAME = "F3"
    IM_SIZE = 65
    N_CLASSES = 2
    RESOLUTION = 1
    # Inline, crossline, timeslice or full
    SLICE = "inline"
    SLICE_NUM = 339
    # BATCH_SIZE = 2**11
    # BATCH_SIZE = 2**8
    # BATCH_SIZE = 2**15
    BATCH_SIZE = 256
    # BATCH_SIZE = 4050
    # number of parallel data loading workers
    N_WORKERS = 4

    # use distributed scoring+
    if RESOLUTION != 1:
        raise Exception("Currently we only support pixel-level scoring")

    # TODO: move these into parser with arguments
    args = Arguments()
    args.gpu = None
    args.batch_size = BATCH_SIZE
    args.workers = N_WORKERS
    args.rank = 0
    # for distributed world_size is the number of VMs initially, but actually it's the
    # sum of GPUs across VMs
    #args.world_size = ngpus_per_node * args.world_size
    # for one VM, it's just the number of GPUs

    args.n_proc_per_gpu = 2
    args.world_size = N_GPU*args.n_proc_per_gpu
    args.dist_url = "tcp://127.0.0.1:12345"
    args.dist_backend = "nccl"
    #args.dist_backend = "gloo"
    args.seed = 0
    args.batch_size = BATCH_SIZE
    args.im_size = IM_SIZE
    args.dataset_name = DATASET_NAME
    args.n_classes = N_CLASSES

    # fix away any kind of randomness - although for scoring it should not matter
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    print("RESOLUTION {}".format(RESOLUTION))

    ##########################################################################

    # Log to tensorboard
    #logger = tb_logger.TBLogger("log", "Test")
    #logger.log_images(
    #    SLICE + "_" + str(SLICE_NUM),
    #    get_slice(data, data_info, SLICE, SLICE_NUM),
    #    cm="gray",
    #)

    # unroll full cube
    indices = []
    predictions = []

    print("-- scoring on GPU --")

    ngpus_per_node = torch.cuda.device_count()
    print("nGPUs per node", ngpus_per_node)

    # do our own spawning
    #manager = Manager()
    #return_dict = manager.dict()
    # return_dict = [None]*N_GPU
    # result_queue = multiprocessing.Queue()
    # mp.spawn(main_worker, nprocs=args.world_size, args=(ngpus_per_node, args))
    """
    with Manager() as manager:
        results_list = manager.list()
        mp.spawn(main_worker, nprocs=args.world_size, args=(ngpus_per_node, results_list, args))
        results = deepcopy(results_list)
    """
    #spawn_context = mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, return_dict, args), join = False)
    #while not spawn_context.join():
    #    pass


    mp = multiprocessing.get_context('spawn')
    # error_queues = []
    processes = []
    for i in range(args.world_size):
        # error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=main_worker,
            args=(i, ngpus_per_node, args),
            daemon=False,
        )
        process.start()
        # error_queues.append(error_queue)
        processes.append(process)

    for process in processes:
        process.join()

    # Loop on join until it returns True or raises an exception.
    # while not spawn_context.join():



    """
    processes = []

    for gpu in range(ngpus_per_node):
        print ("starting thread ", gpu)
        p = Process(target=main_worker, args=(gpu, ngpus_per_node, return_dict, args))
        p.start()
        processes += [p]

    for proc in processes:
        proc.join()

    print (return_dict)
    """
    print("-- aggregating results --")


    # Read 3D cube
    data, data_info = readSEGY(join(args.dataset_name, "data.segy"))

    classified_cube = np.zeros(data.shape)

    # Get half window size
    window = args.im_size // 2
    nx, ny, nz = data.shape

    # generate full list of coordinates
    # memory footprint of this isn't large yet, so not need to wrap as a generator
    #x_list = range(window, nx - window)
    #y_list = range(window, ny - window)
    #z_list = range(window, nz - window)

    #print("-- generating coord list --")
    # TODO: is there any way to use a generator with pyTorch data loader?
    #coord_list = list(itertools.product(x_list, y_list, z_list))
    # coord_list = np.array_split(np.array(coord_list), args.world_size)

    # coord_list = [tuple(x) for x in list(coord_list[args.rank])]

    #print (results)

    new_coord_list = []
    predictions = []
    for i in range(args.world_size):

        with open("results_{}.pkl".format(i), "r") as f:
            dict = json.load(f)

        new_coord_list += dict['coords']
        predictions += dict['preds']

    # assert(coord_list==new_coord_list)

    def worker(classified_cube, coord):
        x, y, z = coord
        ind = new_coord_list.index(coord)
        pred_class = predictions[ind]
        classified_cube[x, y, z] = pred_class

    # launch workers in parallel with memory sharing ("threading" backend)
    _ = Parallel(n_jobs=NUM_CORES, backend="threading")(
        delayed(worker)(classified_cube, coord) for coord in tqdm(new_coord_list)
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
        cm="binary",
    )


if __name__ == "__main__":
    main()

"""
print("-- aggregating results --")

classified_cube = np.zeros(data.shape)

def worker(classified_cube, ind):
    x, y, z = coord_list[ind]
    pred_class = predictions[ind][0][0][0][0]
    classified_cube[x, y, z] = pred_class

# launch workers in parallel with memory sharing ("threading" backend)
_ = Parallel(n_jobs=NUM_CORES, backend="threading")(
    delayed(worker)(classified_cube, ind) for ind in tqdm(indices)
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
    cm="binary",
)
"""

