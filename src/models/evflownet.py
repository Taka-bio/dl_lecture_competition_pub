import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)
        
        #(batch, 2, 32, 32) -> (batch, 2, 256, 256)
        self.deconv1 = nn.ConvTranspose2d(2, 2, kernel_size=8, stride=8, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        #(batch, 2, 64, 64) -> (batch, 2, 256, 256)
        self.deconv2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=4, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        #(batch, 2, 128, 128) -> (batch, 2, 256, 256)
        self.deconv3 = nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_up_0 = self.deconv1(flow.clone())
        flow_dict['flow0'] = flow_up_0

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_up_1 = self.deconv2(flow.clone())
        flow_dict['flow1'] = flow_up_1

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_up_2 = self.deconv3(flow.clone())
        flow_dict['flow2'] = flow_up_2

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        
        return flow_dict
        

if __name__ == "__main__":
    from omegaconf import OmegaConf
    # from config import configs
    import time
    # from data_loader import EventData
    args = OmegaConf.load('configs/base.yaml')
    model = EVFlowNet(args.train).cuda()
    input_ = torch.rand(8,4,256,256).cuda()
    a = time.time()
    output = model(input_)
    b = time.time()
    print(b-a)
    print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
    print(model.state_dict().keys())
    print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)