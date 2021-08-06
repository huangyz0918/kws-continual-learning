import time

def readlines(datapath):
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def sample_dataset(dataloader):
    sample = 0
    start = time.time()
    for index, data in enumerate(dataloader):
        sample = data
        if index == 0:
            break  
    print("batch sampling time:  ", time.time() - start)
    return sample