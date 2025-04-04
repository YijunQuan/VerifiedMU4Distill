import torch
import os
from dataset import dataset_init
from torch.utils.data import DataLoader, random_split
import timeit
from numpy.random import randint
from utils import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset


def teacher_evaluation(dataset = 'MNIST',nt=8, num_epochs=120, 
                save_root_path="./models/"):
    torch.manual_seed(42)
    result_root_path = "./results/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    save_path = save_root_path + dataset
    result_path = result_root_path + dataset

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    _, testset, teacher_constituents, _, batch_size = dataset_init(dataset=dataset, device=device, nt=nt)

    model_weights = torch.load(save_path + f"/teacher_{nt}_shards_{num_epochs}_epochs.pth")
    for i, model in enumerate(teacher_constituents):
        model.load_state_dict(model_weights[i])
    print("All teacher constituents loaded successfully")

    for i, model in enumerate(teacher_constituents):
        
        for param in model.parameters():
            param.requires_grad = False
    print("All teacher weights being fixed")
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    print("Evaluating Teacher Network")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
        
            outputs = 0
            for i, model in enumerate(teacher_constituents):
                outputs += model(images)
        
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Teacher network accuracy is {accuracy}%")

    with open(result_path + f'/teacher_{nt}_{num_epochs}_epochs.pth', "w") as f:
        f.write(f"{accuracy:.2f}%\n")
        print(f"Result saved")


def purge_evaluation(dataset = 'MNIST',nt=8, ns=2, num_epochs=120, 
                save_root_path="./models/", student_percentage=10,
                multi_teacher=True, num_slices = 4):
    torch.manual_seed(42)
    result_root_path = "./results/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    save_path = save_root_path + dataset
    purge_save_path = save_path + '/purge'
    result_path = result_root_path + dataset

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    _, testset, teacher_constituents, student_constituents, batch_size = dataset_init(dataset=dataset, device=device, nt=nt)

    model_weights = torch.load(save_path + f"/teacher_{nt}_shards_{num_epochs}_epochs.pth")
    for i, model in enumerate(teacher_constituents):
        model.load_state_dict(model_weights[i])
    print("All teacher constituents loaded successfully")

    if multi_teacher:
        model_weights = torch.load(purge_save_path + f"/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_{num_slices}_slices.pth")
    else:
        model_weights = torch.load(purge_save_path + f"/single_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_{num_slices}_slices.pth")

    for i, model in enumerate(student_constituents):
        model.load_state_dict(model_weights[i])
    print("All student constituents loaded successfully")


    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
    correct = 0
    total = 0
    print("Evaluating Student Network")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
        
            outputs = 0
            for i, model in enumerate(student_constituents):
                outputs += model(images)
        
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Student network accuracy is {accuracy}%")
    if multi_teacher:
        with open(result_path + f'/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth', "w") as f:
            f.write(f"{accuracy:.2f}%\n")
            print(f"PURGE Result Saved")


    else:
        with open(result_path + f'/single_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth', "w") as f:
            f.write(f"{accuracy:.2f}%\n")
            print(f"Single-Teacher Soft Label Result Saved")


def sisa_evaluation(dataset = 'MNIST',nt=8, ns=2, num_epochs=120, 
                save_root_path="./models/", student_percentage=10,
                multi_teacher=True, num_slices = 4):
    
    torch.manual_seed(42)
    result_root_path = "./results/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    save_path = save_root_path + dataset
    sisa_save_path = save_path + '/sisa'
    result_path = result_root_path + dataset

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    _, testset, teacher_constituents, student_constituents, batch_size = dataset_init(dataset=dataset, device=device, nt=nt)

    model_weights = torch.load(save_path + f"/teacher_{nt}_shards_{num_epochs}_epochs.pth")
    for i, model in enumerate(teacher_constituents):
        model.load_state_dict(model_weights[i])
    print("All teacher constituents loaded successfully")


    model_weights = torch.load(sisa_save_path + f"/student_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth")
    for i, model in enumerate(student_constituents):
        model.load_state_dict(model_weights[i])
    print("All student constituents loaded successfully")


    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
    correct = 0
    total = 0
    print("Evaluating Student Network")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
        
            outputs = 0
            for i, model in enumerate(student_constituents):
                outputs += model(images)
        
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Student network accuracy is {accuracy}%")
    with open(result_path + f'/sisa_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.pth', "w") as f:
        f.write(f"{accuracy:.2f}%\n")
        print(f"SISA Result Saved")


def purge_time_simulation(dataset = 'MNIST', num_rounds = 100, nt=8, ns=2, num_epochs=120, 
                batch_size=32, learning_rate_std=1e-2, 
                save_root_path="./models/", student_percentage = 10, 
                multi_teacher = True, num_slices=4):
    
    # simulating teacher updates
    torch.manual_seed(42)
    result_root_path = "./results_time/"
    save_path = save_root_path + dataset
    purge_save_path = save_root_path + dataset + '/purge'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, _, teacher_constituents, student_constituents, batch_size = dataset_init(dataset=dataset, device=device, nt=nt, ns=ns)
    model_weights = torch.load(save_path + f"/teacher_{nt}_shards_{num_epochs}_epochs.pth")
    for i, model in enumerate(teacher_constituents):
        model.load_state_dict(model_weights[i])
    print("All teacher constituents loaded successfully")


    # fix teacher
    for i, teacher in enumerate(teacher_constituents):
        for param in teacher.parameters():
            param.requires_grad = False
    print("All teacher weights being fixed")
    
    result_path = result_root_path + dataset
    if not os.path.isdir(result_root_path):
        os.mkdir(result_root_path)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)


    
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
    # kl_div_loss = nn.KLDivLoss(reduction='batchmean')


    # number of chunks (nt/ns)

    teacher_ensembles = teacher_allocate(teacher_constituents, ns)
    # simulating a teacher change with equal chance to all teachers. No change in student data
    for round in range(num_rounds):
        # determine which teacher constituent is changed
        student_id = randint(ns)
        current_shard = student_shards[student_id]
        current_teachers = teacher_ensembles[student_id]
        nc = len(current_teachers)
        num_epochs_students = num_epochs * 2//(nc*num_slices+1)
        if (num_epochs_students*2) % (nc*num_slices+1) > 0: num_epochs_students+=1
        print(f"\nLearning from {nc} teachers\n")
        shard_size = len(current_shard)
        chunk_size = shard_size // nc
        sizes = [chunk_size] * nc
        sizes[-1] += shard_size % nc
        chunk_set = []
        start = 0
        
        chunk_id = randint(nc)
        for size in sizes:
            chunk = Subset(current_shard, range(start, start+size))
            start = start + size
            chunk_set.append(chunk)
    
        sliced_chunk = chunk_slice(chunk_set, num_slices)

        # chunk_loaders = [DataLoader(chunk, batch_size=batch_size, shuffle=False) for chunk in chunk_set]
        print(chunk_id)
        model = student_constituents[student_id]

        if chunk_id == 0:
            # first one reset to random initialization
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            if multi_teacher:
                model_weight = torch.load(purge_save_path + f'/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_model{student_id}_chunk{chunk_id}_slice{num_slices-1}.pth')
            else:
                model_weight = torch.load(purge_save_path + f'/single_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_model{student_id}_chunk{chunk_id}_slice{num_slices-1}.pth')

            model.load_state_dict(model_weight)
        for param in model.parameters():
            param.requires_grad = True
        model.train()
        # retraining for the unlearning process
        optimizer = optim.Adam(model.parameters(), lr=learning_rate_std)

        # Update Soft Labels
        soft_labels_chunk = [[None for _ in range(num_slices)] for _ in range(len(sliced_chunk))]  # Initialize structure

        for j in range(chunk_id*num_slices,nc*num_slices, 1):
            for k in range(j+1):
                current_chunk_id = k // num_slices
                current_slice_id = k % num_slices
                slice = sliced_chunk[current_chunk_id][current_slice_id]
                slice_loader = DataLoader(slice, batch_size=batch_size, shuffle=False)
                soft_labels_list = []  # Temporary storage for soft labels of the current slice
                for images, labels in slice_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    if multi_teacher:
                        soft_labels = 0
                        for m in range(current_chunk_id+1):
                            teacher = current_teachers[m]
                            soft_labels += teacher(images)
                        soft_labels = soft_labels / float(k+1)
                    else:
                        teacher = current_teachers[current_chunk_id]
                        soft_labels = teacher(images)
                    soft_labels_list.append(soft_labels)  # Store soft labels for the current batch
                soft_labels_chunk[current_chunk_id][current_slice_id] = soft_labels_list  # Store soft labels for the slice

        start_time = timeit.default_timer()

        for j in range(chunk_id*num_slices,nc*num_slices, 1):
            for k in range(j+1):
                current_chunk_id = k // num_slices
                current_slice_id = k % num_slices
                for epoch in range(num_epochs_students):
                    slice = sliced_chunk[current_chunk_id][current_slice_id]
                    slice_loader = DataLoader(slice, batch_size=batch_size, shuffle=False)
                    soft_labels_slice = soft_labels_chunk[current_chunk_id][current_slice_id]  # Get soft labels for the current slice
                    running_loss = 0.0
                    for batch_idx, (images, _) in enumerate(slice_loader):  # Use ith shard
                        images = images.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        soft_labels = soft_labels_slice[batch_idx]  # Retrieve precomputed soft labels
                        soft_labels = soft_labels.to(device)  # Move to the same device as the model
                        loss = mse_loss(outputs, soft_labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    print(f"Model {student_id+1} - Epoch [{epoch+1}/{num_epochs_students}]-Chunk [{current_chunk_id+1}]-Div[{current_slice_id+1}], Loss: {running_loss / len(slice_loader):.4f}")
        elapsed = timeit.default_timer() - start_time
        if multi_teacher:
            with open(result_path + f'/purge_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_slice_{num_slices}.csv', 'a') as f:
                f.write(f'{elapsed}\n')
        else:
            with open(result_path + f'/single_student_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent_slice_{num_slices}.csv', 'a') as f:
                f.write(f'{elapsed}\n')
        f.close()
        print(f'Round [{round+1}/{num_rounds}] Retraining Time: {elapsed} seconds')


def sisa_time_simulation(dataset = 'MNIST', nt=8, ns=8, num_epochs=120,
        batch_size=64, learning_rate_std=2e-3, save_root_path="./models/",    
            student_percentage = 10, num_rounds = 10):
    
    torch.manual_seed(42)
    result_root_path = "./results_time/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    save_path = save_root_path + dataset
    result_path = result_root_path + dataset
    result_path_baseline = result_path + '/baseline'
    

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    if not os.path.isdir(result_path_baseline):
        os.mkdir(result_path_baseline)

    trainset, testset, teacher_constituents, student_constituents, batch_size = dataset_init(dataset=dataset, device=device, nt=nt, ns=ns)

    model_weights = torch.load(save_path + f"/teacher_{nt}_shards_{num_epochs}_epochs.pth")
    for i, model in enumerate(teacher_constituents):
        model.load_state_dict(model_weights[i])
    print("All teacher constituents loaded successfully")

    for i, model in enumerate(teacher_constituents):
        for param in model.parameters():
            param.requires_grad = False

    print("All teacher weights being fixed")

    for round in range(num_rounds):
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


        # tea_shard_loaders = [DataLoader(subset, batch_size=64, shuffle=True) for subset in teacher_shards]
        # Loss function and optimizer
        # criterion = nn.MSELoss()
        
        mse_loss = nn.MSELoss()
        # number of chunks (nt/ns)
        total_time = 0
        for i, model in enumerate(student_constituents):
            print(f"\nTraining Model {i+1}/{ns} on Shard {i+1}...\n")
            model.train()
            shard_loader = shard_loaders[i]
            optimizer = optim.Adam(model.parameters(), lr=learning_rate_std)
            soft_labels_list = []  # Temporary storage for soft labels of the current slice
            for images, _ in shard_loader:
                for teacher in teacher_constituents:
                    images = images.to(device)
                    soft_labels = teacher(images)
                soft_labels = soft_labels / len(teacher_constituents)
                soft_labels_list.append(soft_labels)  # Store soft labels for the current batch
            # record the time after the soft label generation
            start_time = timeit.default_timer()
            for epoch in range(num_epochs):
                running_loss = 0.0
                for batch_idx, (images, _) in enumerate(shard_loader):  # Use ith shard
                    soft_labels = soft_labels_list[batch_idx]  # Retrieve precomputed soft labels
                    optimizer.zero_grad()

                    # for teacher in teacher_constituents:
                    #     images = images.to(device)
                    #     soft_labels += teacher(images)
                    # soft_labels = soft_labels / len(teacher_constituents)

                    # dummy soft label for time test
                    teacher = teacher_constituents[0]
                    images = images.to(device)
                    outputs = model(images)
                    # loss = criterion(outputs, soft_labels)
                    loss = mse_loss(outputs, soft_labels)

        
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Model {i+1} - Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(shard_loader):.4f}")

            print(f"Finished Training Student Model {i+1}\n")
            total_time += timeit.default_timer() - start_time
        with open(result_path + f'/sisa_{nt}_tshards_{ns}_shards_{num_epochs}_epochs_{student_percentage}_percent.csv', 'a') as f:
                f.write(f'{total_time}\n')
        f.close()
        print(f'Round [{round+1}/{num_rounds}] Retraining Time: {total_time} seconds')
