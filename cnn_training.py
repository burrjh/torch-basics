import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# load model architectures from file in folder
from architectures import CNN, three_layer_net

# let's define two classical model architectures first
# --------------------------------------------------------------------------------

class three_layer_net(nn.Module): 
    ''' the most basic three layered neurel net'''
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(three_layer_net, self).__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,n_classes),
        )

    def forward(self,x):
        x = torch.flatten(x,1) # flatten inputs to 1D, but leaving batch dimension
        logits = self.architecture(x) # forward pass through 3 layers
        return logits # outputs logits, note that softmax might be applied afterwards


class CNN(nn.Module):
    ''' classical Convolutional architecture'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)    # convolutional operation
        self.pool  = nn.MaxPool2d(2,2)   # pooling
        self.conv2 = nn.Conv2d(6,16,5)   
        self.fc1 = nn.Linear(16*5*5,100) # fully connected layer
        self.fc2 = nn.Linear(100,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))  # conv followed by nonlinear relu then pooling
        x = self.pool(F.relu(self.conv2(x)))  # again
        x = torch.flatten(x,1)  # flatten to 1D
        x = F.relu(self.fc1(x)) # fully connected layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
        
if __name__ == '__main__': 
    # this ensures that loop is only executed if file is called directly
    # generally considered good practices, but necessary for torch's dataloader 

        
    # load model
    #-----------------------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # automatically use gpu if cuda is installed
    # later only one model will be trained, you can change it here
    #model = three_layer_net(32*32*3,1000,10).to(device)  # will be easier on your cpu but perform alot worse
    model = CNN().to(device)
    # in case you already trained it for some epochs load it here
    #model.load_state_dict(torch.load('./trained_model'))

    # define hyperparameters
    # ------------------------------------------------------
    epochs = 10
    batchsize = 32 # should always be a multiple of 16 because of the way gpus work
    learning_rate = 1e-3 # determine step size
    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)


    # load the dataset
    # ---------------------------------------------------------------------------------

    # transform images to normalized [-1,1]
    transformation = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
    # loading from torch library, change to download=True when executing the first time
    train_data = torchvision.datasets.CIFAR10(root='./data',train=True,download=False, transform=transformation)
    #test_data  = torchvision.datasets.CIFAR10(root='./data',train=False,download=False, transform=transformation)

    # cifar10 contains 32x32 colored images of these types
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #for training we need a dataloader, which provides the batches for each step
    train_loader = torch.utils.data.DataLoader(train_data,
                                batch_size=batchsize,
                                shuffle=True,         # should be shuffled during training to provide more randomness
                                num_workers=2)        # hard to say what's optimal here, on windows it needs to be 0 zero sometimes



    # training loop
    # -------------------------------------------------------------

    losses_list = []

    for epoch in range(epochs):

        print("Starting Epoch ", epoch+1)
        running_loss = 0.0

        for i,batch in enumerate(train_loader): 
            # split up the batch 
            input,label = batch
            input, label = input.to(device),label.to(device) # manually move it to gpu

            # never forget to null the gradients
            optimizer.zero_grad() 

            # forward pass
            output = model(input)
            loss = loss_fct(output,label)
            running_loss += loss
            
            # backpropagation: all the magic in 2 lines
            loss.backward()
            optimizer.step()

            # print statistics every 1000 steps
            if i % 200 == 199:
                print("Loss: ", running_loss.item()/200)
                losses_list.append(running_loss.item())
                running_loss = 0.0


    # save loss development so we can plot it later
    import pickle
    with open('losses.pkl', 'wb') as f:
        pickle.dump(losses_list, f)

    # save the trained model for later usage
    torch.save(model.state_dict(), './trained_model')






