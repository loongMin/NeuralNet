f"""
# convolution net model:
## layers i:
layers shape: [width, height, channel_i]


## kernels i:
+ convolution:
    kernels shape: [width, height, channel_(i-1), channel_i]
    forward: Z^[i] = W^[i] * Z^[i-1], 
    backward: dW^[i] += dZ^[i] * Z^[i-1], dZ^[i-1] += W^[i] * dZ^[i]
    
+ pooling:
    max        
    mean
             
+ full connected layers:
    


+ input layer 0: [width, height, 3(rgb)]

+ convolution output layer:

## Programming Parameters
+ kernels: 
    convolution: (layers, kernels, kernel_width, kernel_height, channel_i-1)
    pooling: max, mean
+ padding
+ step
+ layers: (layers, layer_width, layer_height, channel)
+ 






"""


















class ConvNet:



    def create_net(self):
        """

        :return:
        """