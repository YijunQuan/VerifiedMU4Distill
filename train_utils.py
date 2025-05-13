import torch
from torch.utils.data import random_split, DataLoader, Subset
from utils import teacher_allocate, chunk_slice, chunk_slice_label
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import pickle

def train_teacher(teacher_constituents, trainset, nt, batch_size, num_epochs, learning_rate, device, file_path_teacher,dataset):
    total_size = len(trainset)
    subset_size = total_size // nt
    sizes = [subset_size] * nt
    sizes[-1] += total_size % nt
    teacher_shards = random_split(trainset, sizes)
    teacher_shard_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in teacher_shards]
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    for i, model in enumerate(teacher_constituents):
        print(f"\nTraining Model {i+1}/{nt} on Subset {i+1}...\n")
    
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
        
            for data in teacher_shard_loaders[i]:  # Use ith shard
                if dataset == 'sst5':
                       batch = {k: v.to(device) for k, v in data.items()}
                else:
                    images, labels = data
                    inputs, labels = images.to(device), labels.to(device)
                    images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                if dataset == 'sst5':
                    outputs = model(**batch)
                    loss = outputs.loss
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
            print(f"Model {i+1} - Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(teacher_shard_loaders[i]):.4f}")

        print(f"Finished Training Teacher Model {i+1}\n")
        optimizer.zero_grad(set_to_none=True)

        for param in model.parameters():
            param.requires_grad = False

    model_weights = [model.state_dict() for model in teacher_constituents]
    torch.save(model_weights, file_path_teacher)
    print(f"All teacher constituents trained and saved successfully!")


def train_student_purge(student_constituents, teacher_constituents, nt, ns, trainset, student_percentage, num_slices,
                  learning_rate_std, batch_size, num_epochs, device, multi_teacher, purge_save_path, dataset):
    total_size = len(trainset)
    train_size = total_size * student_percentage // 100
    ignored_size = total_size - train_size
    student_train_set, _ = random_split(trainset, [train_size, ignored_size])
    student_total_size = len(student_train_set) 
    subset_size = student_total_size // ns
    sizes = [subset_size] * ns
    sizes[-1] += student_total_size % ns
    student_shards = random_split(student_train_set, sizes)

    # Loss function and optimizer
    mse_loss = nn.MSELoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    # number of chunks (nt/ns)

    teacher_ensembles = teacher_allocate(teacher_constituents, ns)
    for i, model in enumerate(student_constituents):
        print(f"\nTraining Model {i+1}/{ns} on Shard {i+1}...\n")
        
        # divide each shard further to chunks
        current_shard = student_shards[i]
        current_teachers = teacher_ensembles[i]
        nc = len(current_teachers)
        num_epochs_students = num_epochs * 2//(nc+1)
        if num_epochs_students == 0: num_epochs_students=1
        print(f"\nLearning from {nc} teachers\n")
        shard_size = len(current_shard)
        chunk_size = shard_size // nc
        sizes = [chunk_size] * nc
        sizes[-1] += shard_size % nc
        chunk_set = []
        start = 0

        
        for size in sizes:
            chunk = Subset(current_shard, range(start, start+size))
            start = start + size
            chunk_set.append(chunk)
        chunk_loaders = [DataLoader(chunk, batch_size=batch_size, shuffle=False) for chunk in chunk_set]

        optimizer = optim.Adam(model.parameters(), lr=learning_rate_std)
        if multi_teacher:
            # generate soft labels for each chunk
            soft_labels_chunk = [None for _ in range(len(current_teachers))]  # Initialize structure
            for j in tqdm(range(len(current_teachers))):
                soft_labels_list = []
                chunk_loader = chunk_loaders[j]
                for _, data in enumerate(chunk_loader):  # Use ith shard
                    if dataset == 'sst5':
                        batch = {k: v.to(device) for k, v in data.items()}
                    else:
                        images, labels = data
                        inputs, labels = images.to(device), labels.to(device)
                    soft_labels = 0
                    for teacher in current_teachers[:j+1]:
                        teacher.to(device)
                        if dataset == 'sst5':
                            soft_labels += teacher(**batch).logits
                        else:
                            soft_labels += teacher(inputs)
                    soft_labels_list.append(soft_labels / float(j+1))
                soft_labels_chunk[j] = soft_labels_list

        sliced_chunk = chunk_slice(chunk_set, num_slices)
        if multi_teacher:
            soft_labels_slice = chunk_slice_label(soft_labels_chunk, num_slices)
        for j in range(len(current_teachers)*num_slices):
            for k in range((j+1)):
                chunk_id = k // num_slices
                slice_id = k % num_slices
                for epoch in range(num_epochs_students):
                    slice = sliced_chunk[chunk_id][slice_id]
                    if multi_teacher:
                        slice_labels = soft_labels_slice[chunk_id][slice_id]
                    slice_loader = DataLoader(slice, batch_size=batch_size, shuffle=False)
                    
                    model.train()
                    running_loss = 0.0
                    for batch_idx, data in enumerate(slice_loader):  # Use ith shar
                        if dataset == 'sst5':
                            batch = {k: v.to(device) for k, v in data.items()}
                        else:
                            images, labels = data
                            inputs, labels = images.to(device), labels.to(device)

                        if dataset == 'sst5':
                            outputs = model(**batch).logits
                        else:
                            outputs = model(inputs)
                            
                        if multi_teacher:
                            soft_labels = slice_labels[batch_idx]
                        else:
                            teacher = current_teachers[k]
                            if dataset == 'sst5':
                                soft_labels = teacher(**batch).logits
                            else:
                                soft_labels = teacher(images)

                        loss = mse_loss(outputs, soft_labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    print(f"Model {i+1} - Epoch [{epoch+1}/{num_epochs_students}]-Chunk [{chunk_id+1}]-Div[{slice_id+1}], Loss: {running_loss / len(slice_loader):.4f}")

            torch.save(model.state_dict(), purge_save_path + f"/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_model{i}_chunk{chunk_id}_slice{slice_id}.pth")
            print(f"Finished Chunk {chunk_id+1} Slice {slice_id+1}\n")

        print(f"Finished Training Student Model {i+1}\n")
        optimizer.zero_grad(set_to_none=True)

    model_weights = [model.state_dict() for model in student_constituents]
    if multi_teacher:
        torch.save(model_weights, purge_save_path + f"/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_{num_slices}_slices.pth")
    else:
        torch.save(model_weights, purge_save_path + f"/singe_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_{num_slices}_slices.pth")



def train_student_sisa(student_constituents, teacher_constituents, nt, ns, trainset, student_percentage,
                  learning_rate_std, batch_size, num_epochs, device, sisa_save_path, dataset):
    total_size = len(trainset)
    total_size = len(trainset)
    train_size = total_size * student_percentage // 100
    ignored_size = total_size - train_size
    trainsetloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=64)


   

    # Loss function and optimizer    
    mse_loss = nn.MSELoss()

    soft_labels_root = './softlabels'
    if not os.path.exists(soft_labels_root):
        os.makedirs(soft_labels_root)
    if not os.path.exists(soft_labels_root + f"/{dataset}"):
        os.makedirs(soft_labels_root + f"/{dataset}")
    if not os.path.exists(soft_labels_root + f"/{dataset}/sisa"):
        os.makedirs(soft_labels_root + f"/{dataset}/sisa")
    soft_labels_file = "./softlabels/"+dataset +f"/sisa/teacher_{nt}_shards_{num_epochs}_epochs.pkl"

    if not os.path.exists(soft_labels_file):
        soft_labels_list = []
        print(f"\nGenerating Soft Labels for Baseline...\n")
        # if dataset == 'sst5':
        #     for i, model in enumerate(teacher_constituents):
        #         model.to(device)
        for data in tqdm(trainsetloader):
            if dataset == 'sst5':
                batch = {k: v.to(device) for k, v in data.items()}
                labels = batch['labels']
            else:
                images, labels = data
                inputs, labels = images.to(device), labels.to(device)

            soft_labels = 0
            for teacher in teacher_constituents:
                if dataset == 'sst5':
                    soft_labels += teacher(**batch).logits
                else:
                    soft_labels += teacher(inputs)
            soft_labels_list.append(soft_labels/ len(teacher_constituents))
        
        soft_labels_list = [item for sublist in soft_labels_list for item in sublist]
        with open(soft_labels_file, "wb") as f:
            pickle.dump(soft_labels_list, f)
        print(f"Soft labels generated and saved successfully!")
    else:
        with open(soft_labels_file, "rb") as f:
            soft_labels_list = pickle.load(f)
        print(f"Soft labels loaded successfully!")
    # number of chunks (nt/ns)

    # Split the dataset into shards 
    student_train_set, _ = random_split(trainset, [train_size, ignored_size])
    train_indices = student_train_set.indices

    soft_labels_train = [soft_labels_list[i] for i in train_indices]
    student_total_size = len(student_train_set) 
    subset_size = student_total_size // ns
    sizes = [subset_size] * ns
    sizes[-1] += student_total_size % ns
    student_shards = random_split(student_train_set, sizes)
    soft_labels_shards = [
        [soft_labels_train[i] for i in shard.indices] for shard in student_shards
    ]
    shard_loaders = [DataLoader(shard, batch_size=batch_size, shuffle=False, num_workers=64) for shard in student_shards]

    # number of chunks (nt/ns)

    for i, model in enumerate(student_constituents):
        print(f"\nTraining Model {i+1}/{ns} on Shard {i+1}...\n")
        model.train()
        shard_loader = shard_loaders[i]
        soft_labels_shard = soft_labels_shards[i]
        soft_labels_shard_batched = []
        start_idx = 0
        for batch in shard_loader:
            tmp_batch_size = batch['input_ids'].size(0) if dataset == 'sst5' else batch[0].size(0)
            soft_labels_shard_batched.append(soft_labels_shard[start_idx:start_idx + tmp_batch_size])
            start_idx += tmp_batch_size
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_std)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, data in enumerate(shard_loader):  # Use ith shard
                if dataset == 'sst5':
                    batch = {k: v.to(device) for k, v in data.items()}
                    labels = batch['labels']
                else:
                    images, labels = data
                    inputs, labels = images.to(device), labels.to(device)

                soft_labels = torch.stack(soft_labels_shard_batched[batch_idx]).to(device)
                optimizer.zero_grad()

                if dataset == 'sst5':
                    outputs = model(**batch).logits
                    loss = mse_loss(outputs, soft_labels)
                else:
                    outputs = model(inputs)
                    loss = mse_loss(outputs, soft_labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Model {i+1} - Epoch [{epoch+1}/{num_epochs}]], Loss: {running_loss / len(shard_loader):.4f}")

        print(f"Finished Training Student Model {i+1}\n")

    model_weights = [model.state_dict() for model in student_constituents]
    torch.save(model_weights, sisa_save_path + f"/student_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth")
    print(f"All SISA student constituents trained and saved successfully!")

