import torch
import os
def check_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available")

def set_device(gpu_id=0):
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def check_cudnn():
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else "N/A"
    print(f"cuDNN available: {cudnn_available}")
    print(f"cuDNN version: {cudnn_version}")

if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using GPU {device}: {torch.cuda.get_device_name(device)}")
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
    check_available_gpus()
    check_cudnn()
    device = set_device(gpu_id=0)  # Change gpu_id to the desired GPU index