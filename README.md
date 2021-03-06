# FedReN
This is the code for paper :  **Federated Learning on Non-IID and Globally Long-Tailed Data via Meta Re-Weighting Networks**.

**Abstract**: Under a federated learning environment, the training samples are generally collected and stored locally on each client's device, which makes a subsegment  machine learning procedure  not meet the requirement of independent and identical distribution (IID). Existing federated learning methods to deal with non-IID data generally assume that the data is globally balanced. However, real-world multi-class data tends to exhibit long-tail distribution, where the majority of samples are in few head classes and a large number of tail classes only have a small amount of data. This paper, therefore, focuses on addressing the problem of handling non-IID and globally long-tailed data in a federated learning scenario. Accordingly, we propose a new federated learning method called Federated meta Re-weighting Networks (FedReN), which assigns weights during the local training process from the class-level and instance-level perspectives, respectively. To deal with data non-IIDness and global long-tail, both of the two re-weighting functions are globally trained by the meta-learning approach to acquire the knowledge of global long-tail distribution. Experiments on several long-tailed image classification benchmarks show that FedReN outperforms the state-of-the-art federated learning methods. The code is available at https://github.com/pxqian/FedReN.

## Dependencies

* PyTorch >= 1.0.0

* torchvision >= 0.2.1

  

## Parameters

| Parameter     | Description                                              |
| ------------- | -------------------------------------------------------- |
| `dataset`     | Dataset to use. Options: `cifar10`,`cifar100`, `imagenet`. |
| `lr`          | Learning rate of model.                                  |
| `v_lr`        | Learning rate of the class re-weighting network.                   |
| `s_lr`        | Learning rate of the instance re-weighting network.                   |
| `local_bs`    | Local batch size of training.                            |
| `test_bs`     | Test batch size .                                        |
| `num_users`   | Number of clients.                                       |
| `frac`        | the fraction of clients to be sampled in each round.     |
| `epochs`      | Number of communication rounds.                          |
| `local_ep`    | Number of local epochs.                                  |
| `imb_factor`  | Imbalanced control. Options: `0.01`,`0.02`, `0.1`.       |
| `num_classes` | Number of classes.                                       |
| `num_meta`    | Number of meta data per class.                           |
| `embedding_dim`    | dimension of emdedding vector  per class.                           |
| `device`      | Specify the device to run the program.                   |
| `seed`        | The initial seed.                                        |


## Usage

Here is an example to run FedARN on CIFAR-10 with imb_fartor=0.01:

```
python main_fedren.py --dataset=cifar10 \
    --lr=0.01 \
    --v_lr=0.01\
    --s_lr=0.0001\
    --epochs=200\
    --local_ep=5 \
    --local_bs=64\
    --num_users=20 \
    --frac=0.5\
    --num_meta=10 \
    --num_classes=10 \
    --imb_factor=0.01\
    --embedding_dim=64\
```

