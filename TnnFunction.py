#importing libraries 
import torch
from torch.autograd import Function



class Timing_function_for(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight, bias=None,Q=1):
        #genrate array of sorted index
        input_sorted,sorted_index= torch.sort(input)
        sort= torch.zeros_like(sorted_index[0])


        output=torch.zeros(input.shape[0],weight.shape[0])
        contri_weight=torch.zeros(input.shape[0],weight.shape[0],weight.shape[1])
        for i in range(input_sorted.shape[0]):
             sort[sorted_index[i]]=torch.arange(0,input_sorted.shape[1])
             #print(input[i],sort)
             #generate n new arrays with boolean mask from sorted_indices
             weight_broadcast = torch.where(sort.unsqueeze(0) <=torch.arange(0,sort.shape[0]).unsqueeze(1),weight.unsqueeze(1) ,torch.zeros_like(weight).unsqueeze(1))
             #print(weight_broadcast)
             #change weight_broadcast to float
             weight_broadcast = weight_broadcast.float()
             #perform matrix multiplication of input and weight_broadcast
             all_output = ((Q)+ torch.tensordot(input[i],weight_broadcast, dims=([0],[2])))/weight_broadcast.sum(dim=2)
             #print(all_output)
             #find the min of all_output
             min_values,min_indices = torch.min(all_output, dim=1)
            #  print(min_values)
            #  print(min_indices)
             output[i] = min_values
             contri_weight[i] = weight_broadcast[torch.arange(weight_broadcast.size(0)), min_indices].unsqueeze(0)
 
        # The forward pass can use ctx.
        ctx.save_for_backward(input, contri_weight, bias, output)
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        #print(contri_weight)

        return output
 
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_input = grad_weight = grad_bias=Q = None
        # print(input)
        # print('weight', weight)
        # print('grad_output', grad_output)
        # print('output', output)
        if ctx.needs_input_grad[0]:
            grad_input = torch.tensordot(grad_output,(weight/weight.sum(dim=2,keepdim=True)), dims=([1],[1]))
            #print(grad_input)
            grad_input = grad_input[torch.eye(weight.shape[0]).bool()]
        if ctx.needs_input_grad[1]:
            grad_weight = (input.unsqueeze(1) - output.unsqueeze(2))/weight.sum(dim=2,keepdim=True) 
            #print(grad_weight)
            grad_weight =torch.tensordot(grad_output,(((input.unsqueeze(1) - output.unsqueeze(2))/weight.sum(dim=2,keepdim=True))* (weight !=0).float()), dims=([0],[0]))
            grad_weight = grad_weight[torch.eye(weight.shape[1]).bool()]
            #print(grad_weight)
        # # if bias is not None and ctx.needs_input_grad[2]:
        # #     grad_bias = grad_output.sum(0)
        # print(weight.sum(dim=2,keepdim=True))
        print(grad_input)
        print(grad_weight)
        return grad_input, grad_weight, grad_bias,Q
    

class Timing_function(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight, bias=None, Q=1):
        #genrate array of sorted index
        input_sorted,sorted_index= torch.sort(input)
        sort= torch.zeros_like(sorted_index)

        sort.scatter_(1,sorted_index,torch.arange(0,input_sorted.shape[1]).repeat(input_sorted.shape[0],1))
        #print(input,sort)
        #generate n new arrays with boolean mask from sorted_indices
        weight_broadcast = torch.where(sort.unsqueeze(1).unsqueeze(1) <=torch.arange(0, sort.shape[1]).unsqueeze(0).unsqueeze(2).unsqueeze(1),weight.unsqueeze(1).unsqueeze(0) ,torch.zeros_like(weight).unsqueeze(1).unsqueeze(0))
        weight_broadcast= weight_broadcast.float()
        # print("weight",weight_broadcast)
        #perform matrix multiplication of input and weight_broadcast
        diagonal = torch.tensordot(input,weight_broadcast, dims=([1],[3]))
        diagonal = diagonal[torch.eye(weight_broadcast.shape[0]).bool()]
        all_output =((Q)+diagonal)/weight_broadcast.sum(dim=3)
        # print(all_output)
        #find the min of all_output
        min_values,min_indices = torch.min(all_output, dim=2)
        # print(min_values)
        #print(min_indices)
        output = min_values
        contri_weight = weight_broadcast[torch.repeat_interleave(torch.arange(weight_broadcast.size(0)),weight_broadcast.size(1)),torch.arange(weight_broadcast.size(1)).repeat(1,weight_broadcast.size(0)),min_indices.view(-1)].reshape(weight_broadcast.shape[0],weight_broadcast.shape[1],weight_broadcast.shape[2])
   
        # The forward pass can use ctx.
        ctx.save_for_backward(input, contri_weight, bias,output)
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        return output
 
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_input = grad_weight = grad_bias =Q= None
        # print(input)
        # print('weight', weight)
        # print('grad_output', grad_output)
        # print('output', output)
        if ctx.needs_input_grad[0]:
            grad_input = torch.tensordot(grad_output,(weight/weight.sum(dim=2,keepdim=True)), dims=([1],[1]))
            #print(grad_input)
            grad_input = grad_input[torch.eye(weight.shape[0]).bool()]
        if ctx.needs_input_grad[1]:
            grad_weight = (input.unsqueeze(1) - output.unsqueeze(2))/weight.sum(dim=2,keepdim=True) 
            #print(grad_weight)
            grad_weight =torch.tensordot(grad_output,(((input.unsqueeze(1) - output.unsqueeze(2))/weight.sum(dim=2,keepdim=True))* (weight !=0).float()), dims=([0],[0]))
            grad_weight = grad_weight[torch.eye(weight.shape[1]).bool()]
            #print(grad_weight)
        # # if bias is not None and ctx.needs_input_grad[2]:
        # #     grad_bias = grad_output.sum(0)
        # print(weight.sum(dim=2,keepdim=True))
        #print(grad_input)
        #print(grad_weight)
        return grad_input, grad_weight, grad_bias, Q
    

