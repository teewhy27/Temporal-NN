#importing libraries 
import numpy as np
import torch
import TnnFunction as tnn
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import DownsampleDataloader as Dl
import TnnModel as tnnM
import argparse


# to run on GPU for M1 activate virtual environment with:  conda activate torch-nightly 
# torch.device("mps") analogous to torch.device("cuda") on an Nvidia GPU.
import torch.nn as nn




if __name__ == "__main__":

    print("performing training on one neuron model")

    parser = argparse.ArgumentParser(description="Training variables for one neuron model")

    # Define the arguments
    parser.add_argument("--learning_rate" , type=float, help="learning rate for the trainer", default=0.5)
    parser.add_argument("--epochs" , type=int, help="number of epochs for the trainer", default=100)
    parser.add_argument("--batch_size" , type=int, help="batch size for the trainer", default=100)
    parser.add_argument("--subset" , type=bool, help="subset of the dataset", default=False)
    parser.add_argument("--subset_indices" , type=int, help="subset indices of the dataset", default=1000)
    parser.add_argument("--Q" , type=int, help="Q for the trainer", default=10)
    parser.add_argument("--input_features" , type=int, help="input features for the trainer", default=10)
    parser.add_argument("--output_features" , type=int, help="output features for the trainer", default=1)
    parser.add_argument("--bias" , type=bool, help="bias for the trainer", default=False)
    parser.add_argument("--device" , type=str, help="device for the trainer", default="cpu")
    parser.add_argument("--loss_function" , type=str, help="loss function for the trainer", default="MSELoss")
    parser.add_argument("--optimizer" , type=str, help="optimizer for the trainer", default="SGD")
    parser.add_argument("--time_scale" , type=float, help="time scale of the output", default=2)


    # Parse the arguments
    args = parser.parse_args()
    learning_rate=args.learning_rate
    num_epochs=args.epochs
    batch_size=args.batch_size
    subset=args.subset
    subset_indices=args.subset_indices
    Q=args.Q
    input_features=args.input_features
    output_features=args.output_features
    bias=args.bias
    device=args.device
    time_scale=args.time_scale
    #model for 1 output neuron
    one_neuron_model=tnnM.Temporal(10,1,False,Q=10).to(device)
    #setup loss function and optimizer
    if args.loss_function=="MSELoss":
        criterion = nn.MSELoss()
    elif args.loss_function=="CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        #raise error
        print("loss function not defined")
        print("Add loss function to allowed arguments and try again")

    if args.optimizer=="SGD":
        optimizer = torch.optim.SGD(one_neuron_model.parameters(), lr=learning_rate)
    elif args.optimizer=="Adam":
        optimizer = torch.optim.Adam(one_neuron_model.parameters(), lr=learning_rate)
    else:
        #raise error
        print("optimizer not defined")
        print("Add optimizer to allowed arguments and try again")

    #device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    train_loader = Dl.train_loader(batch_size=100,subset=True,subset_indices=1000)



    # setup the output label for one neuron model
    one_labels=torch.ones(batch_size).long()
    # Train the model
    total_step = len(train_loader)

    #save weight before training
    weight_before_training=one_neuron_model.weight1.data.clone()

    #create one random input between zero and one and duplicate it to create a batch
    one_input=torch.rand(1,10)
    one_input=one_input.repeat(batch_size,1)

    #save weight after each epoch
    one_neuron_model_weight_after_epoch=[]
    # We use the pre-defined number of epochs to determine how many iterations to train the network on
    for epoch in range(num_epochs):
        #Load in the data in batches using the train_loader object
        for i in range(1):
            # Move tensors to the configured device
            images = one_input.to(device)
            #images = images.type(torch.FloatTensor) # convert the images to float tensors used when running on CPU
            labels = one_labels.to(device) #set the labels to 1

            #convert the labels to one hot vectors and add the time scale to the labels
            #labels_hot=one_hot_batch(labels)
            labels_hot=labels+time_scale
            labels_hot=labels_hot.float()
            #print(labels_hot)
            
            # Forward pass
            #flatted the image to a vector of size 100
            #images= images.view(images.size(0), -1)
            outputs = one_neuron_model(images) # Pass the images to the CNN model. images contains the batch size of 10 images and the rest of the dimensions are inferred by the CNN
            loss = criterion(outputs, labels_hot) # Calculate the loss using the loss function criterion which is the cross entropy loss function

            #compute the accuracy of the model
            _, predicted = torch.max(outputs.data, 1) # torch.max returns the maximum value and the index of the maximum value in the tensor. The index of the maximum value is the predicted class
            #print("outputs",outputs)
            #print("predicted",predicted)
            #print("labels",labels)
            correct = (predicted == labels).sum().item() # sum up the number of correct predictions and convert the tensor to a scalar value
            accuracy = correct / labels.size(0) # divide the number of correct predictions by the batch size to get the accuracy of the model
            # the difference between accuracy and loss is that accuracy is the number of correct predictions divided by the batch size and loss is the average loss of the batch
            # Backward and optimize
            optimizer.zero_grad() # zero out the gradients from the previous iteration
            loss.backward() # backpropagate the loss computes the gradients of the loss with respect to the parameters of the model the backward function is called on and take grad_output as the argument which is the gradient of the loss with respect to the output of the model and is computed by the loss function
            optimizer.step() # update the parameters of the model using the gradients computed by the backward function

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        print('Accuracy: {:.2f}%'.format(accuracy*100))
        one_neuron_model_weight_after_epoch.append(one_neuron_model.weight1.data.clone())
    



 