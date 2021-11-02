import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import sys
sys.path.append('raft')
from raft.raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import numpy as np

from flowtools import warp, fbcCheckTorch
from torchvision.utils import save_image

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for TCL')
            parser.add_argument('--lambda_TCL', type=float, default=10.0, help='weight for VGG loss')
            parser.add_argument('--lambda_c3D', type=float, default=10.0, help='weight for 3D cycle loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', "VGG_A", "VGG_B", "TCL_A", "TCL_B", "c3D_A", "c3D_B"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'real_A2', 'fake_B2', 'rec_A2', 'warp_B', 'fuse_B']#, 'mask_A'
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B2', 'fake_A2', 'rec_B2', 'warp_A', 'fuse_A']#, 'mask_B'
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'F_A', 'F_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netF_A = networks.define_F(opt.init_type, opt.init_gain, self.gpu_ids)#.to("cuda")
        self.netF_B = networks.define_F(opt.init_type, opt.init_gain, self.gpu_ids)#.to("cuda")
        #self.netVGG_19 = networks.Vgg19().to("cuda")
        self.raftModel = self.initRaftModel(opt)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF_A.parameters(), self.netF_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)

    def initRaftModel(self, opt):
      model = torch.nn.DataParallel(RAFT(opt))
      model.load_state_dict(torch.load("raft/models/raft-chairs.pth"))
  
      model = model.module
      model.to('cuda')
      model.eval()
      
      return model

    def computeRAFT(self, img1, img2, it=20):
      with torch.no_grad():
        padder = InputPadder(img1.shape)
        image1, image2 = padder.pad(img1, img2)
        flow_low, flow_up = self.raftModel(image1, image2, iters=it, test_mode=True)
        
      return flow_up
    
    def generateMask(self, simg, wimg):
      return torch.exp(-50*torch.abs(simg - wimg).mean())

    def set_input_fc2(self, data):
        img1, img2, simg1, simg2 = data
        
        self.real_A = img1.to(self.device)
        self.real_A2 = img2.to(self.device)
        self.real_B = simg1.to(self.device)
        self.real_B2 = simg2.to(self.device)
        
    def set_input(self, inp):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = inp['A' if AtoB else 'B'].to(self.device)
        self.real_B = inp['B' if AtoB else 'A'].to(self.device)
        self.image_paths = inp['A_paths' if AtoB else 'B_paths']

    def forward_train(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # 1st frame (t-1)
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)  XT-1 
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        
        # 2nd frame (t)
        self.fake_B2 = self.netG_A(self.real_A2)  # G_A(A)
        self.rec_A2 = self.netG_B(self.fake_B2)   # G_B(G_A(A))
        self.fake_A2 = self.netG_B(self.real_B2)  # G_B(B)
        self.rec_B2 = self.netG_A(self.fake_A2)   # G_A(G_B(B))

        self.bf_real_A = self.computeRAFT(self.real_A2, self.real_A)
        self.warp_B = warp(self.fake_B, self.bf_real_A)
        self.fuse_B = self.netF_A(self.fake_B2, self.warp_B)  #XT
        self.mask_A = self.generateMask(self.real_A2, warp(self.real_A, self.bf_real_A))
        self.vgg_fuse_B = self.netVGG_19(self.fuse_B)
        self.vgg_real_A2 = self.netVGG_19(self.real_A2)
        
        self.bf_fake_B = self.computeRAFT(self.fuse_B, self.fake_B)
        self.rec3D_A2 = self.netF_B(self.netG_B(self.fuse_B), warp(self.fake_B, self.bf_fake_B))
        
        self.bf_real_B = self.computeRAFT(self.real_B2, self.real_B)
        self.warp_A = warp(self.fake_A, self.bf_real_B)
        self.fuse_A = self.netF_B(self.fake_A2, self.warp_A)
        self.mask_B = self.generateMask(self.real_B2, warp(self.real_B, self.bf_real_B))
        self.vgg_fuse_A = self.netVGG_19(self.fuse_A)
        self.vgg_real_B2 = self.netVGG_19(self.real_B2)
        
        self.bf_fake_A = self.computeRAFT(self.fuse_A, self.fake_A)
        self.rec3D_B2 = self.netF_A(self.netG_A(self.fuse_A), warp(self.fake_A, self.bf_fake_A))
        
        '''
        save_image(((self.real_A[0]+ 1)/ 2.0).detach().cpu(), 'im1_real_A.png')
        save_image(((self.real_A2[0]+ 1)/ 2.0).detach().cpu(), 'im1_real_A2.png')
        save_image(((self.fake_B[0]+ 1)/ 2.0).detach().cpu(), 'im2_fake_B.png')
        save_image(((self.rec_A[0] + 1)/ 2.0).detach().cpu(), 'im3_rec_A.png')
        save_image(((self.warp_B[0]+ 1)/ 2.0).detach().cpu(), 'im4_warp_B.png')
        save_image(((self.fuse_B[0]+ 1)/ 2.0).detach().cpu(), 'im5_fuse_B.png')
        save_image(((self.mask_A[0]+ 1)/ 2.0).detach().cpu(), 'im6_mask_A.png')
        
        blah
        '''
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        
    def forward_eval(self, inp, wrp=None, AtoB=True):
      img = inp.to(self.device)
      
      with torch.no_grad():
        if AtoB:
          sty = self.netG_A(img)
        else:
          sty = self.netG_B(img)
        
        if wrp == None:
          return sty
        
        if AtoB:
          fused = self.netF_A(sty, wrp)
        else:
          fused = self.netF_B(sty, wrp)
      return fused

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_VGG = self.opt.lambda_VGG
        lambda_TCL = self.opt.lambda_TCL
        lambda_c3D = self.opt.lambda_c3D
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        self.loss_c3D_A = self.criterionCycle(self.rec3D_A2, self.real_A2) * lambda_c3D
        self.loss_c3D_B = self.criterionCycle(self.rec3D_B2, self.real_B2) * lambda_c3D
        
        self.loss_VGG_A = 0#(((self.vgg_fuse_B - self.vgg_real_A2)**2)**0.5).mean() * lambda_VGG
        self.loss_VGG_B = 0#(((self.vgg_fuse_A - self.vgg_real_B2)**2)**0.5).mean() * lambda_VGG
        
        self.loss_TCL_A = (self.mask_A*torch.abs(self.fuse_B - self.warp_B)).mean() * lambda_TCL
        self.loss_TCL_B = 0#(self.mask_B*torch.abs(self.fuse_A - self.warp_A)).mean() * lambda_TCL

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_VGG_A + self.loss_VGG_B + self.loss_TCL_A + self.loss_TCL_B + self.loss_c3D_A + self.loss_c3D_B
        self.loss_G.backward()#retain_graph=True

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        
        # forward
        #self.forward()      # compute fake images and reconstruction images.
        self.forward_train()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_F.step()       
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        
        #with open('losses.txt', 'a') as log_file:
        #  #log_file.write(self.get_current_losses() + '\n')
        #  print(self.get_current_losses(), file=log_file)

