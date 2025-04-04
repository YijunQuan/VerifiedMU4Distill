import torch
from torch.utils.data import random_split, DataLoader, Subset
from utils import teacher_allocate, chunk_slice
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_teacher(teacher_constituents, trainset, nt, batch_size, num_epochs, learning_rate, device, file_path_teacher):
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
        
            for images, labels in teacher_shard_loaders[i]:  # Use ith shard
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
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
                  learning_rate_std, batch_size, num_epochs, device, multi_teacher, purge_save_path):
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
        num_epochs_students = int(np.ceil(num_epochs * 2/(nc*num_slices+1)))

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
        sliced_chunk = chunk_slice(chunk_set, num_slices)
        # chunk_loaders = [DataLoader(chunk, batch_size=batch_size, shuffle=False) for chunk in chunk_set]

        optimizer = optim.Adam(model.parameters(), lr=learning_rate_std)
        for j in range(len(current_teachers)*num_slices):
            for k in range((j+1)):
                chunk_id = k // num_slices
                slice_id = k % num_slices
                for epoch in range(num_epochs_students):
                    slice = sliced_chunk[chunk_id][slice_id]
                    slice_loader = DataLoader(slice, batch_size=batch_size, shuffle=False)
                    model.train()
                    running_loss = 0.0
    
                    for images, labels in slice_loader:  # Use ith shard
                        images = images.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        if multi_teacher:
                            soft_labels = 0
                            for m in range(chunk_id+1):
                                teacher = current_teachers[m]
                                soft_labels += teacher(images)
                            soft_labels = soft_labels / float(chunk_id+1)
                            loss = mse_loss(outputs, soft_labels)
                        else:
                            teacher = current_teachers[k]
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
                  learning_rate_std, batch_size, num_epochs, device, sisa_save_path):
    total_size = len(trainset)
    total_size = len(trainset)
    train_size = total_size * student_percentage // 100
    ignored_size = total_size - train_size
    student_train_set, _ = random_split(trainset, [train_size, ignored_size])
    student_total_size = len(student_train_set) 
    subset_size = student_total_size // ns
    sizes = [subset_size] * ns
    sizes[-1] += student_total_size % ns
    student_shards = random_split(student_train_set, sizes)
    shard_loaders = [DataLoader(shard, batch_size=batch_size, shuffle=False) for shard in student_shards]


    # Loss function and optimizer    
    mse_loss = nn.MSELoss()
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')


    # number of chunks (nt/ns)

    for i, model in enumerate(student_constituents):
        print(f"\nTraining Model {i+1}/{ns} on Shard {i+1}...\n")
        model.train()
        shard_loader = shard_loaders[i]
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_std)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, _ in shard_loader:  # Use ith shard
                soft_labels = 0
                optimizer.zero_grad()

                for teacher in teacher_constituents:
                    images = images.to(device)
                    soft_labels += teacher(images)
                
                soft_labels = soft_labels / len(teacher_constituents)

                outputs = model(images)
                # loss = criterion(outputs, soft_labels)
                loss = mse_loss(outputs, soft_labels)

    
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Model {i+1} - Epoch [{epoch+1}/{num_epochs}]], Loss: {running_loss / len(shard_loader):.4f}")

        print(f"Finished Training Student Model {i+1}\n")

    model_weights = [model.state_dict() for model in student_constituents]
    torch.save(model_weights, sisa_save_path + f"/student_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth")

    print(f"All SISA student constituents trained and saved successfully!")

