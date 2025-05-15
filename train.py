import os
import torch
from dataset import dataset_init
from train_utils import *
import argparse

def purge_train(dataset = 'MNIST',nt =8, ns=2, num_epochs=120, 
                batch_size=2, learning_rate=0.001, learning_rate_std=5e-3, 
                save_root_path="./models/", student_percentage = 10,
                multi_teacher = True, num_slices=4, device = "cuda"):

    # nt: number of teacher constituents
    # ns: number of student constituents
    # Fixed the random seed to repeat shard splitting
    # multi_teacher for PURGE, otherwise single teacher soft-label
    purge_save_path = save_root_path + dataset + '/purge'
    sisa_save_path = save_root_path + dataset + '/sisa'
    teacher_save_path = save_root_path + dataset

    if not os.path.isdir(teacher_save_path):
        os.mkdir(teacher_save_path)
    
    if not os.path.isdir(purge_save_path):
        os.mkdir(purge_save_path)

    if not os.path.isdir(sisa_save_path):
        os.mkdir(sisa_save_path)

    file_path_teacher = teacher_save_path + f"/teacher_{nt}_shards_{num_epochs}_epochs.pth"

    if multi_teacher:
        file_path_student_purge = purge_save_path + f"/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_{num_slices}_slices.pth"

    else:
        file_path_student_purge = purge_save_path + f"/single_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_{num_slices}_slices.pth"


    file_path_student_sisa = sisa_save_path + f"/student_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth"


    saved_teacher = os.path.isfile(file_path_teacher)
    saved_student_purge = os.path.isfile(file_path_student_purge)
    saved_student_sisa = os.path.isfile(file_path_student_sisa)
   
    trainset, _, teacher_constituents, student_constituents, batch_size = dataset_init(dataset=dataset, device=device, nt=nt, ns=ns)

    # teacher training or loading (For demonstration purpose, teacher is trained with 1 slice per shard)
    if not saved_teacher:
        train_teacher(teacher_constituents, trainset, nt, batch_size, num_epochs, learning_rate, device, file_path_teacher, dataset=dataset)
    else:
        model_weights = torch.load(file_path_teacher)
        for i, model in enumerate(teacher_constituents):
            model.load_state_dict(model_weights[i])
        print("All teacher constituents loaded successfully")

    for i, teacher in enumerate(teacher_constituents):
        for param in teacher.parameters():
            param.requires_grad = False
    print("All teacher weights being fixed")
    

    # student constituents trainig or loading
    if not saved_student_purge:
        train_student_purge(student_constituents, teacher_constituents, nt, ns, trainset, student_percentage, num_slices,
                  learning_rate_std, batch_size, num_epochs, device, multi_teacher, purge_save_path, dataset=dataset)
        

    if not saved_student_sisa:
        train_student_sisa(student_constituents, teacher_constituents, nt, ns, trainset, student_percentage,
                  learning_rate_std, batch_size, num_epochs, device, sisa_save_path, dataset=dataset)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distillation Unlearning Training")
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (default: MNIST)')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'MNIST':
        batch_size = 512
        learning_rate_std = 5e-3
        learning_rate = 0.001
    elif dataset == 'SVHN':
        batch_size = 512
        learning_rate_std = 2e-3
        learning_rate = 2e-3
    elif dataset == 'sst5':
        batch_size = 32
        learning_rate_std = 5e-5
        learning_rate = 2e-5
    elif dataset == 'cifar100':
        batch_size = 32
        learning_rate_std = 2e-3
        learning_rate = 2e-3
    else:
        raise Exception("Dataset Not Supported")

    torch.manual_seed(42)
    if not os.path.isdir('./models/'):
        os.mkdir('./models/')
    

    purge_train(dataset=dataset, batch_size=batch_size, num_epochs=1, nt=4, ns=2, student_percentage=10,learning_rate=learning_rate, learning_rate_std=5e-5)
    