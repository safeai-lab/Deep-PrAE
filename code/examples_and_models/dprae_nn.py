#dprae nn

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)        
        
        return out
    
    
def trainer(train_input, train_label, batch_size = 1000, n_iters = 1000, 
            log=False, it_interval=100, input_dim=2, hidden_dim=5, output_dim=2):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_dataset = []
    for i in range(train_label.shape[0]): 
        train_dataset.append([train_input[i,:], train_label[i]])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    input_dim = input_dim
    hidden_dim = hidden_dim
    output_dim = output_dim
    
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    net = model(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    #weights
    class_weight = torch.Tensor([0.1, 1]).to(device)
    sample_weight = torch.ones(batch_size).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='none')


    it = 1
    if log == True: print("Logging set as true:")
        
    for epoch in range(num_epochs):
        for i, (data, target) in enumerate(train_loader):
            it = it + 1
            X = Variable(torch.Tensor(data.float())).to(device)
            y = Variable(torch.Tensor(target.float())).to(device)      
            optimizer.zero_grad()        
            net_out = net(X)
            loss = criterion(net_out, y.long())
            loss = loss * sample_weight #considers weighted loss
            loss.mean().backward()
            optimizer.step()

            if log == True:
                if it % it_interval == 0:
                    err_fp = (np.logical_and(net_out.max(1)[1].data.to("cpu") == 1, y.long().to("cpu")==0)).float().mean()
                    err_fn = (np.logical_and(net_out.max(1)[1].data.to("cpu") == 0, y.long().to("cpu")==1)).float().mean()
                    print(epoch+1, it, "Loss: {0:0.3f}".format(loss.mean().data.cpu().numpy()), "FP: {:0.3f}".format(err_fp.data.numpy()), "FN: {:0.3f}".format(err_fn.data.numpy()), end="\r")
                    
    return net.eval()    