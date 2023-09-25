import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from copy import deepcopy
import os
import re
import yaml
import random
from scipy.ndimage import zoom
from config import ALLOWED_METRICS, ALLOWED_MODELS, ALLOWED_SCHEDULERS
from torch.utils.data import DataLoader
from training_config import training_params
from utils import find_experiment_index
from dataset import balance_dataset

class Model:
    def __init__(self, model_name : str, save_path : str, pretrained=True):
        if model_name not in ALLOWED_MODELS:
            raise NotImplementedError(f'Model {model_name} is not supported yet')
        # load model
        self.model = getattr(torchvision.models, model_name)(pretrained=True)
        if model_name == 'resnet18':
            self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=1000, bias=True),
                                          nn.Linear(in_features=1000, out_features=1, bias=True),
                                          nn.Sigmoid()       
                                         )
        elif model_name in [f'efficientnet_b{i}' for i in range(8)]:
            self.model.classifier =  nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                   nn.Linear(in_features=self.model.classifier[1].in_features, out_features=1),
                                                   nn.Sigmoid()
                                                  )
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_name = model_name
        self.fitted = False
        self.experiment_index = find_experiment_index(save_path)
        self.save_dir = save_path
        self.run_name = f'run{self.experiment_index}'
        self.experiment_path = f'{self.save_dir}/{self.run_name}'
        
    # load weights of model
    def load(self, run_name=None, best=True):
        if not run_name:
            run_name = self.run_name
        else:
            self.run_name = run_name
            self.experiment_path = f'{self.save_dir}/{self.run_name}'
        if best:
            self.model.load_state_dict(torch.load(f'{self.save_dir}/{run_name}/best'))
        else:
            self.model.load_state_dict(torch.load(f'{self.save_dir}/{run_name}/last'))
        self.fitted = True
    
    def isfitted(self):
        return self.fitted
    
    # freeze feature extractor
    def freeze_backbone(self):
        if model_name in [f'efficientnet_b{i}' for i in range(8)]:
            for param in self.model.features.parameters():
                param.requires_grad = False
        elif model_name == 'resnet18':
            for name, param in model.named_parameters():
                if name == 'fc':
                    break
                param.requires_grad = False
       
    # gives parameters of model to optimization algorithms
    def get_model_params(self):
        return self.model.parameters()
    
    # show learning plots
    def save_training_plots(self):
        try:
            data = pd.read_csv(f'{self.experiment_path}/training_logs.csv', index_col=0)
            x_data = data['epoch'].tolist()
            
            # all columns have names like training_NAME or valid_NAME
            # so to get acutal column names we need to extract NAMES
            raw_columns = list(data.drop(columns=['epoch']).columns)
            names = ['_'.join(col.split('_')[1:]) for col in raw_columns]
            
            # save only unique names
            names = set(names)
            n_rows = int(len(names)/2)
            n_cols = 2
            figure, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(16, 10))
            
            for i, name in enumerate(names):
                current_df = data.filter(regex=f'.*{name}$')
                df_col_names = list(current_df.columns)
                
                j, k = i//n_cols, i%n_cols
                axs[j][k].set_title(name)
                    
                y_data = current_df.iloc[:, 0].tolist()
                axs[j][k].plot(x_data, y_data, color='black', label=f'{df_col_names[0]}')
                y_data = current_df.iloc[:, 1].tolist()
                axs[j][k].plot(x_data, y_data, color='red',label=f'{df_col_names[1]}')
            plt.legend()
            plt.savefig(f'{self.experiment_path}/learning_curves.png')
            plt.close()
        except FileNotFoundError:
            print("Please, check csv file with training loss and training data and path to it")
            
    def class_activation_map(self, img, target=None):
        if not re.search(r'efficientnet', self.model_name):
            raise NotImplementedError('Function can support efficientnet architectures only in that stage of project')
        img = torch.tensor(img)
        img_tensor = img.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        output = self.model(img_tensor)
        conv_output = self.model.features(img_tensor)    # output of model before pooling
        
        prediction = 1 if output.item() >= 0.5 else 0
            
        # extract weigts of the last layer before pooling
        w = 0
        for layer in list(self.model.named_parameters()):
            if layer[0] == 'features.8.1.weight':
                w = layer[1].cpu().detach().numpy()
            
        # acuire heatmap
        conv = conv_output.squeeze().cpu().detach().numpy()
        heatmap = 0
        for i in range(conv.shape[0]):
            heatmap += w[i]*conv[i]
        
        scale = 512 / 16
        plt.figure(figsize=(12, 12))
        
        img = img.detach().numpy()
        res = np.moveaxis(img, 0, 2)
        plt.imshow(res[:,:,::-1])
        plt.imshow(zoom(heatmap, zoom=(scale, scale)), cmap='jet', alpha=0.5)
        plt.savefig(f'{self.experiment_path}/heatmap.png')
        print(f"Target is: {target}", f"Prediction is: {prediction}", sep='\n')
        
    # currently, only for batch_size=1.
    def predict_proba(self, inputs):
        outputs = []
        with torch.no_grad():
            for elem in inputs:
                # elem = elem.float()
                elem = np.expand_dims(elem, axis=0)
                elem = torch.tensor(elem).to(self.device)
                prediction = self.model(elem).to('cpu').detach().item()
                outputs.append(prediction)
        return np.array(outputs)
    
    def predict(self, inputs):
        probs = self.predict_proba(inputs)
        return np.where(probs > 0.5, 1, 0)
        
    def random_class_activation_map(self, dataset):
        idx = random.randrange(0, len(dataset))
        img, target = dataset[idx]
        self.class_activation_map(img, target)           
        
    def __run_epoch(self, dataloader, metrics, criterion, optimizer=None, group='inference', threshold=0.5) -> dict:
        # main variables
        running_loss = 0.0
        y_true = []
        y_pred = []
        
        def dataloader_loop():
            nonlocal y_true, y_pred, running_loss, optimizer, criterion
            for inputs, labels in dataloader:
                if group == 'train':
                    optimizer.zero_grad()
                    
                # move the input and model to GPU for speed if available
                inputs = inputs.float()
                labels = labels.unsqueeze(1)
                labels = labels.float()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                    
                outputs = self.model(inputs)
                outputs = outputs.to(self.device)
                loss = criterion(outputs, labels)
                
                # add predictions and true labels for general statistics
                y_true.append(labels.to('cpu').detach().numpy())
                y_pred.append(outputs.to('cpu').detach().numpy())
                
                # backpropogating if training
                if group == 'train':
                    loss.backward()
                    optimizer.step()    
                running_loss += loss.item()

        # preliminary preparation for loop
        if group == 'inference':
            self.model.eval()
            with torch.no_grad():
                dataloader_loop()
        elif group == 'train':
            self.model.train()  
            dataloader_loop()
           
        # compute metrics
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        metric_dict = {}
        for _metric in metrics:
            estimator = getattr(sklearn.metrics, _metric)
            value = 0
            if _metric != 'roc_auc_score':
                value = estimator(y_true.round().astype('int16'), 
                                  y_pred.round().astype('int16'))
            else:
                value = estimator(y_true, y_pred)
            metric_dict[_metric] = value

        loss = running_loss/(len(dataloader)*training_params[f'{group}_batch_size'])
        metric_dict['loss'] = loss
        return metric_dict
    
    def inference(self, dataset, sampler=None, metrics=['accuracy_score', 'precision_score', 'roc_auc_score']):
        # form inference dataloader
        generator = DataLoader(dataset=dataset, batch_size=training_params['inference_batch_size'], 
                                     shuffle=False, num_workers=training_params['num_workers'], sampler=sampler)
        criterion = getattr(torch.nn, training_params['loss'])(reduction='sum')
        res_dict = self.__run_epoch(generator, metrics, criterion=criterion, group='inference')
        for key, value in res_dict.items():
            print(f'{key}: {value}')
        return res_dict
    
    def fit(self, train_data, valid_data, metrics=['accuracy_score', 'precision_score', 'roc_auc_score'], 
                selection_criterion='roc_auc_score'):    
        if self.fitted:
            raise RuntimeError(f'The model {self.model_name} has been fitted already. Try to use another model entity')
            
        if selection_criterion not in metrics + ['loss']:
            raise ValueError(f"""Selection criterion {selection_criterion} is not in metrics list. Change criterion to one of
                            {metrics} or to 'loss'. Otherwise define metric explicitly in metrics list.""")
            
        # create directory to save training information
        if not os.path.isdir(self.experiment_path):
            os.mkdir(self.experiment_path)
            
        # we need to save information about learning process
        train_logs = {'epoch': [], 'train_loss': [], 'valid_loss': []}
        for _metric in metrics:
            for sample in ['valid', 'train']:
                train_logs[f'{sample}_{_metric}'] = []
        
        # applying train_generator
        train_generator = []
        if training_params['balanced'] == True:
            weighted_sampler = balance_dataset(train_data)
            train_generator = DataLoader(dataset=train_data, batch_size=training_params['train_batch_size'],
                                         sampler=weighted_sampler, num_workers=training_params['num_workers'])
        else:
            train_generator = DataLoader(dataset=train_data, batch_size=training_params['train_batch_size'], 
                                     shuffle=True, num_workers=training_params['num_workers'])
        valid_generator = DataLoader(dataset=valid_data, batch_size=training_params['inference_batch_size'], 
                                     shuffle=False, num_workers=training_params['num_workers'])
        
        # define optimizer and loss function
        optimizer = getattr(torch.optim, training_params['optimizer'])(self.get_model_params(), lr=training_params['lr'],
                                                                      weight_decay=training_params['wd'])
        criterion = getattr(torch.nn, training_params['loss'])(reduction='sum')
        scheduler = None
        if training_params['scheduler'] and training_params['scheduler'] in ALLOWED_SCHEDULERS:
            init_scheduler = getattr(torch.optim.lr_scheduler, training_params['scheduler'])
            scheduler = init_scheduler(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
        
        best_value = None
        best_value_epoch = None
        
        # savling training params information and additional useful information
        params_logs = deepcopy(training_params)
        params_logs['best_value_epoch'] = best_value_epoch
        params_logs['selection_criterion'] = selection_criterion
        with open(f'{self.experiment_path}/params_logs.yaml', 'w') as out:
            yaml.dump(params_logs, out)
            
        for epoch in range(1, training_params['epochs']+1):
            print(f'Epoch number: {epoch}')       
            # run training process and get data
            train_dict = self.__run_epoch(train_generator, metrics, criterion=criterion,
                                          optimizer=optimizer, group='train')
            valid_dict = self.__run_epoch(valid_generator, metrics, criterion=criterion,
                                          optimizer=optimizer, group='inference')
            scheduler.step(valid_dict['loss'])
                
            # write data to the table
            train_logs['epoch'].append(epoch)
            for key, value in train_dict.items():
                train_logs[f'train_{key}'].append(value)
            for key, value in valid_dict.items():
                train_logs[f'valid_{key}'].append(value)
            
            # print statistics
            print('Training data: ', end='')
            for key, value in train_dict.items():
                print(f'{key} {value:.5f}', end=', ')
            print('\nValid data: ', end='')
            for key, value in valid_dict.items():
                print(f'{key} {value:.5f}', end=', ')
            print()
                
            # check if our model is better than previos based on selection criterion
            current_value = valid_dict[selection_criterion]
            if not best_value:
                best_value = current_value
                best_value_epoch = epoch
            elif (selection_criterion == 'loss' and best_value > current_value) or (best_value < current_value):
                best_value = current_value
                best_value_epoch = epoch
                params_logs['best_value_epoch'] = best_value_epoch
                with open(f'{self.experiment_path}/params_logs.yaml', 'w') as out:
                    yaml.dump(params_logs, out)
                torch.save(self.model.state_dict(), f'{self.experiment_path}/best')
                
            # save last model, logs, training plots and so on
            torch.save(self.model.state_dict(), f'{self.experiment_path}/last')
            tmp = pd.DataFrame(train_logs)
            tmp.to_csv(f'{self.experiment_path}/training_logs.csv')
            self.save_training_plots()
        
        self.fitted = True
        print('Finished Training')