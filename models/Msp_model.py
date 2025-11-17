import torch
from .base_model import BaseModel
from . import networks
#import ttools.modules
from . import MSP
import torch.nn as nn
from . import cm
import itertools

class mspModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(norm='batch', netG='FTGAN_MLAN', dataset_mode='font')
        
        if is_train:
            parser.set_defaults(batch_size=256, pool_size=0, gan_mode='hinge', netD='basic_64')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
            parser.add_argument('--lambda_NCE', type=float, default=2.0, help='weight for content loss')
            parser.add_argument('--lambda_G_NCE', type=float, default=0.5, help='weight for content loss')
            parser.add_argument('--nce_layers', type=str, default='0,1,2', help='compute NCE loss on which layers')
            parser.add_argument('--dis_2', default=False, help='use two discriminators or not')
            parser.add_argument('--use_spectral_norm', default=True)
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
            
        if self.isTrain:
            self.dis_2 = opt.dis_2
            self.visual_names = ['gt_images', 'generated_images']+['style_images_{}'.format(i) for i in range(self.style_channel)]
            
            self.model_names = ['P','P_style']
            self.loss_names = ['nce']
        else:
            self.visual_names = ['gt_images', 'generated_images']
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        

        if self.isTrain:  # define discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            
            style_vgg = MSP.vgg
            #style_vgg.load_state_dict(torch.load('models/style_vgg.pth'))
            style_vgg = nn.Sequential(*list(style_vgg.children()))
            self.netP = MSP.StyleExtractor(style_vgg,self.gpu_ids )
            self.netP_style = MSP.Projector(self.gpu_ids)  
            self.netP = networks.init_net(self.netP, 'normal', 0.02, self.gpu_ids) 
            self.netP_style = networks.init_net(self.netP_style, 'normal', 0.02, self.gpu_ids)
            self.memory = cm.ClusterMemory(2080,818,momentum=0.1)
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
            self.lambda_NCE = opt.lambda_NCE
            self.optimizer_MSP = torch.optim.Adam(itertools.chain(self.netP.parameters(), self.netP_style.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_MSP)

      
            
    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        self.cids = data['char_label']
        self.fids = data['font_label'] 
        if not self.isTrain:
            self.image_paths = data['image_paths']
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    

        
    def backward_NCEloss(self):
        real_gt = self.netP((self.gt_images),self.nce_layers) 
        query = self.netP_style(real_gt,self.nce_layers)
        num = 0
        self.loss_nce = 0
        for x in self.nce_layers:
            self.loss_nce += self.memory(query[num], self.fids,x,grad=1)
            #self.nce_loss.dequeue_and_enqueue(key[num], K,'real_B{:d}'.format(x))
            num += 1
        
        return self.loss_nce*self.opt.lambda_NCE
    
    def optimize_parameters(self):
        torch.cuda.empty_cache()
        
        # update MSP
        if self.opt.lambda_NCE>0:
           
            self.optimizer_MSP.zero_grad()
            self.loss_NCE = self.backward_NCEloss()
            self.loss_NCE.backward(retain_graph=True)
            self.optimizer_MSP.step()  
        return self.loss_NCE
