
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from helper import *



def train_module(model, optimizer,scheduler, loss_fn, train_loader, val_loader, writer , epochs=20, device="cpu",  path = MODEL_PATH):
    
    
    start_epoch = 1
    best_valid_loss = float('inf')
    BEST_MODEL_='resnet32_Mhana_cifar_bestmodel.pt'
    MODEL_='resnet32_Mhana_cifar_currentmodel.pt'
    path_best_model = os.path.join(path,BEST_MODEL_)
    
    

    #upload the best model if exist and start from there
    losslogger =[]
    model, optimizer,scheduler, start_epoch, losslogger= load_checkpoint(model, optimizer,scheduler, losslogger, filename=path_best_model)
    
    #move the model and the optimizer to the gpu if it exists 
    model.to(device)
    for state in optimizer.state.values():
      for k, v in state.items():
          if isinstance(v, torch.Tensor):
              state[k] = v.to(device)

    for epoch in range(start_epoch, epochs+1):
        
        training_loss = 0.0
        valid_loss = 0.0

        #set up model to training mode
        model.train()
        for i,batch in enumerate(train_loader):
            #set the gradients to zero (because it turns out the gradients are accumlated)
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            #pass data through our model
            output = model(inputs)
            
            #compute loss function
            loss = loss_fn(output, targets)

            #compute the gradients using backpropogation
            loss.backward()

            #update the weights of our model using the computed gradiens
            optimizer.step()

            #accumlate the training loss
            training_loss += loss.data.item() * inputs.size(0)
            
            #logginh training loss
            if i % 1000 == 999:    # every 1000 mini-batches...
                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(model, inputs, targets),
                                global_step=epoch * len(train_loader) + i)
        

        scheduler.step()
        #compute the average of training loss
        training_loss /= len(train_loader.dataset)
        
        #set up model on evaluation mode
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        #save best model
        if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),'scheduler':  scheduler.state_dict(), 'losslogger': losslogger, }
          torch.save(state, path_best_model)
         
        #save model every epoch
        path_model = os.path.join(path,MODEL_)
        best_valid_loss = valid_loss
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),'scheduler':  scheduler.state_dict(), 'losslogger': losslogger, }
        torch.save(state, path_model)
        
         # ...log the training loss
        writer.add_scalar('training_loss',
                        training_loss ,
                        epoch )
        # ...log the valid loss
        writer.add_scalar('valid_loss',
                        valid_loss ,
                        epoch )
        # ...log the accuracy  
        writer.add_scalar('accurracy',
                        num_correct / num_examples ,
                        epoch )
        
        losslogger.append({'epoch':epoch,'valid_loss':valid_loss,'training_loss':training_loss,'accurracy':num_correct / num_examples})
    
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
    writer.flush()
    writer.close()