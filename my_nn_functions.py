import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

import time
import copy
from tqdm import tqdm_notebook

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import defaultdict

from torch.nn.functional import softmax, interpolate


def trainNet(model, n_epochs, optimizer, loss_function, dataloaders, dataset_sizes, device, scheduler=None, upscale=False):
    
    print("===== TRAIN STARTED =====")
   
    n_batches = len(dataloaders['train'])
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    training_start_time = time.time()
    print_every = n_batches // 4 -1
    
    loss_hist = defaultdict(list)
    acc_hist = defaultdict(list)
    
    for epoch in tqdm_notebook(range(n_epochs)):
        
        print(f'Epoch {epoch+1}/{n_epochs}')
        print('-' * 10)
        
        start_time = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0.0
            running_samples = 0
            
            for i, (inputs, labels) in enumerate(tqdm_notebook(dataloaders[phase])):
                inputs = interpolate(inputs, scale_factor =8).to(device) if upscale else inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_samples += inputs.size(0)
                
                if phase == 'train':
                    if (i + 1) % (print_every + 1) == 0:
                        print(f"Epoch {epoch+1}, {100 * (i+1) / n_batches:.2f}% \
 train_loss: {running_loss / running_samples:.3f} \
 train acc: {running_corrects.float() / running_samples:.2f} \
 took: {time.time() - start_time:.2f}s")
                        start_time = time.time()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'Phase: {phase} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.2f}')
            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            
    time_elapsed = time.time() - training_start_time
    print(f'Training complete in {int(time_elapsed) // 60}m {time_elapsed % 60:.2f}s')
    print(f'Best val Acc: {best_acc}')

    model.load_state_dict(best_model_wts)
    
    acc_hist['train'] = torch.stack(acc_hist['train']).cpu().numpy()
    acc_hist['val'] = torch.stack(acc_hist['val']).cpu().numpy()
    loss_hist['train'] = np.array(loss_hist['train'])
    loss_hist['val'] = np.array(loss_hist['val'])
    
    print("===== TRAIN FINISHED =====")
   
    return model, loss_hist, acc_hist

def get_cifar_dataloaders_sizes():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop((32,32), scale=(0.9,1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    image_datasets['train'] = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=data_transforms['train'])

    image_datasets['val'] =  torchvision.datasets.CIFAR10(root='./cifardata', train=False, transform=data_transforms['val'])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def get_preds(model, dataloaders, upscale = False):
    y_true = []
    y_true_pred_proba = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in dataloaders['val']:
                inputs = interpolate(inputs, scale_factor=8).to(device) if upscale else inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                y_true.append(labels)
                y_true_pred_proba.append(softmax(outputs,1))
    y_true = torch.cat(y_true)
    y_true_pred_proba = torch.cat(y_true_pred_proba)
    return y_true.cpu().numpy(), y_true_pred_proba.cpu().numpy()

def class_true_preds(y_true, probas, i):
    y2 = y_true.copy()
    y2[y2 != i] = -1
    y2[y2 == i] = 1
    y2[y2 != 1] = 0
    
    preds = probas[:,i]
    
    return y2, preds

def get_roc(model, dataloaders, class_names, net_name, upscale=False):
    
    y_true, y_true_pred_proba = get_preds(model, dataloaders, upscale)
    
    plt.cla()
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    fig = plt.figure(figsize=(10, 12), dpi= 300, facecolor='w', edgecolor='k')
    for i in range(10):
        y2, preds = class_true_preds(y_true, y_true_pred_proba, i)
        fpr, tpr, threshold = metrics.roc_curve(y2, preds)
        roc_auc = metrics.auc(fpr, tpr)

        plt.subplot(5,2,i+1)
        plt.title(f'ROC {class_names[i]} vs all')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        if i <= 7:
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        if i > 7:
            plt.xlabel('False Positive Rate')
    plt.savefig("plots/" + net_name + '.png')
    plt.show()

def plot_history(hist, crit_name, net_name):
    n_epo = hist['train'].shape[0]
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    plt.figure(figsize=(5, 4), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(1, n_epo+1), hist['train'], 'o-', range(1, n_epo+1), hist['val'], 'o-')
    plt.ylabel(crit_name)
    plt.xlabel("Epoch")
    if crit_name == 'Accuracy':
        leg_labs = [f'train: {np.max(hist["train"]):.3f}', f'val: {np.max(hist["val"]):.3f}']    
    else:
        leg_labs = [f'train: {np.min(hist["train"]):.3f}', f'val: {np.min(hist["val"]):.3f}']
    plt.legend(leg_labs)
    plt.savefig("plots/"+net_name + crit_name +'.png')
    plt.show()
    