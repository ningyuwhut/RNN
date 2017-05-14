#encoding=utf8
import copy, numpy as np
import sys
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {} #数字对应的二进制形式
binary_dim = 8

largest_number = pow(2,binary_dim) #2^8
print "largest_number", largest_number
#解压成对应的二进制形式, 纬度是largest_number*8
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
print "binary"
print binary
print binary.shape
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1 #学习率
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 #纬度(2,16)
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0) #纬度和synapse_0一样的0矩阵
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000): #10000次迭代
    
    #每次迭代随机生成一个样本进行训练，可以看做是sgd,没有batch
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c) #存储每次预测出来的位序列

    overallError = 0
    
    layer_2_deltas = list() #输出层的残差
    layer_1_values = list() 
    layer_1_values.append(np.zeros(hidden_dim)) #这是隐藏层的输出,初始化为0向量,记录每个时刻隐藏层的输出
    #初始为0表示没有隐藏层
    
    #前向传播
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]]) #二进制位是从右往左的
#        print "X", X
#        print "X.shape", X.shape
        y = np.array([[c[binary_dim - position - 1]]]).T #真实值
        print "y", y, y.shape

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h)) #(1,16),隐藏层输出

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1)) #(1,1),标量,输出层

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2 
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2)) #输出层的残差
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1)) #存储隐藏层输出，以便下一时刻进行前向传递
    
    future_layer_1_delta = np.zeros(hidden_dim) #存储下一时刻的隐藏层的残差，以进行后向传播
    
    #后向传播
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1] #当前隐藏层输出
        prev_layer_1 = layer_1_values[-position-2] #前一时刻隐藏层输出
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        #这里的误差是怎么传播的
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
        #相当于损失只向前传播一个时刻，即bptt的t=1

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"
