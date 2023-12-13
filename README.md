# Temporal-NN
20k array working NN code in PyTorch

# DownsampleDataloader
The downdample dataloader helps you select a either the subset or full Mnist dataset, more options can be added if required 
to use 
import DownsampleDataloader as Dl
train_loader = Dl.train_loader(batch_size=100,subset=True,subset_indices=1000)

you can also run Dl.train_loader(batch_size=100) by default subset is false and the entire dataset is used.


# TnnFunction 

Contains the autograd function exteded for the forward and backward path of the Temporal Nueral network 

# TnnModel

Contains the 1 layer Model for the Temporal Nerual network. inmport the python module and specify the input, output and Q value for the network 

# one_layer_tnn 

one layer trainer taking 10 inputs features and 1 output features. output labels have been hardcoded to a single value and input values are random times between zero and one reapeated accross the batch 
possible arguments 
one_layer_tnn.py [--learning_rate LEARNING_RATE] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--subset SUBSET] [--subset_indices SUBSET_INDICES] [--Q Q] [--input_features INPUT_FEATURES]
                        [--output_features OUTPUT_FEATURES] [--bias BIAS] [--device DEVICE] [--loss_function LOSS_FUNCTION] [--optimizer OPTIMIZER] [--time_scale TIME_SCALE]

all have defualt values 
