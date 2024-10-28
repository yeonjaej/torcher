import numpy as np
from torcher.models.resnet import ResNetLayer 
import torch
import torch.nn.functional as F

class UResNet(torch.nn.Module):

    def __init__(self, num_class, num_regress, num_input, num_output_base, blocks, mlp_size=256, bn_momentum=0.9):
        """
        Args: num_class ... integer, number of filters in the last layer
              num_input ... integer, number of channels in the input data
              num_output_base ... integer, number of filters in the first layer
              blocks ... list of integers, number of ResNet modules at each spatial dimensions
         """

        super().__init__()
        
        self._encoder  = []
        self._upsampler= []
        self._decoder  = []
        
        num_output = num_output_base
        features = []
        for block_index, num_modules in enumerate(blocks):

            stride = 2 if block_index > 0 else 1

            self._encoder.append([])
            self._encoder[-1].append( ResNetLayer(num_input, num_output, num_modules, stride=stride, momentum=bn_momentum) )
            self._encoder[-1].append( torch.nn.ReLU(inplace=True) )
            # For the next layer, increase channel count by 2
            features.append((num_input,num_output))
            num_input  = num_output
            num_output = num_output * 2
            
        for i in range(len(features)-1):
            num_output,num_input = features[-1*(i+1)]
            num_modules = blocks[-1*i]
            self._upsampler.append(torch.nn.ConvTranspose2d(num_input,
                                                            num_output,
                                                            3, 
                                                            stride=2, 
                                                            padding=1)
                                  )
            self._decoder.append([])
            self._decoder[-1].append( ResNetLayer(num_output*2, num_output, num_modules, stride=1, momentum=bn_momentum) )
            self._decoder[-1].append( torch.nn.ReLU(inplace=True) )
            self._decoder[-1].append( ResNetLayer(num_output,   num_output, num_modules, stride=1, momentum=bn_momentum) )
            self._decoder[-1].append( torch.nn.ReLU(inplace=True) )

        # Create sequential object to register operations
        self.__unet_ops = []
        for module_lists in [self._encoder,self._decoder]:
            for module_list in module_lists:
                for module in module_list:
                    self.__unet_ops.append(module)
        for module in self._upsampler:
            self.__unet_ops.append(module)
        self.__unet_ops = torch.nn.Sequential(*self.__unet_ops)
            
        self._classifier, self._regressor = None,None


        if num_class:
            classifier=[]
            classifier.append(torch.nn.Conv2d(num_output_base,mlp_size,kernel_size=1,stride=1,bias=True))
            classifier.append(torch.nn.ReLU(inplace=True))
            classifier.append(torch.nn.Conv2d(mlp_size,mlp_size,kernel_size=1,stride=1,bias=True))
            classifier.append(torch.nn.ReLU(inplace=True))
            classifier.append(torch.nn.Conv2d(mlp_size,num_class,kernel_size=1,stride=1,bias=False))

            self._classifier = torch.nn.Sequential(*classifier)

        if num_regress:
            regressor=[]
            regressor.append(torch.nn.Conv2d(num_output_base,mlp_size,kernel_size=1,stride=1,bias=True))
            regressor.append(torch.nn.ReLU(inplace=True))
            regressor.append(torch.nn.Conv2d(mlp_size,mlp_size,kernel_size=1,stride=1,bias=True))
            regressor.append(torch.nn.ReLU(inplace=True))
            regressor.append(torch.nn.Conv2d(mlp_size,num_regress,kernel_size=1,stride=1,bias=False))

            self._regressor = torch.nn.Sequential(*regressor)

    def forward(self,x,show_shape=False):
        
        features = [x]
        if show_shape: print('Input ...',x.shape)
        for i in range(len(self._encoder)):
            data = features[-1]
            for module in self._encoder[i]:
                data = module(data)
            features.append(data)
            if show_shape: print('After encoder block',i,'...',features[-1].shape)
            
        decoder_input = features[-1]
        
        for i in range(len(self._upsampler)):

            decoder_input = self._upsampler[i](decoder_input, output_size=features[-1*(i+2)].size())
            if show_shape: print('After upsample',i,'...',decoder_input.shape)

            decoder_input = torch.cat([decoder_input,features[-1*(i+2)]],dim=1)
            if show_shape: print('After concat  ',i,'...',decoder_input.shape)

            for module in self._decoder[i]:
                decoder_input = module(decoder_input)
            if show_shape: print('After decoder ',i,'...',decoder_input.shape)
        

        pred_class, pred_value = None, None

        result=dict()
        if self._classifier:
            pred_class = self._classifier(decoder_input)
            if show_shape:
                print('Result classifier:',pred_class.shape)
            result['classifier']=pred_class

        if self._regressor:
            pred_value = self._regressor(decoder_input)
            if show_shape:
                print('Result regressor :',pred_value.shape)
            result['regressor']=pred_value

        return result

if __name__ == '__main__':

    num_class  = 3
    num_input  = 1
    num_output_base = 16
    num_modules = [2,2,3,3,3]
    num_regress = 2

    tensor = torch.Tensor(np.zeros(shape=[10,1,256,256],dtype=np.float32)).to('cuda')
    net = UResNet(num_class, num_regress, num_input, num_output_base, num_modules).to('cuda')
    net(tensor,show_shape=True)
