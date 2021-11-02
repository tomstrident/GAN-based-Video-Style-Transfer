from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

from PIL import Image
import shutil
from sg2_core import utils
from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images

from collections import OrderedDict
from tqdm import tqdm

import sintel_eval
from torch.utils import data
from torchvision import transforms
from sintel_eval import initRaftModel, SingleSintelVideo, computeTCL, save_dict_as_json 

def chunks(lst, n):
  #"""Yield successive n-sized chunks from lst."""
  #for i in range(0, len(lst), n):
  #  yield lst[i:i + n]
  return [lst[i:i + n] for i in range(0, len(lst), n)]

def load_image(imfile, device="cuda"):
  img = np.array(Image.open(imfile)).astype(np.uint8)
  img = torch.from_numpy(img).permute(2, 0, 1).float()
  return img[None].to(device)

def load_images(imgs_path):
  imgs_path_list = os.listdir(imgs_path)
  imgs_path_list.sort()
  return [load_image(os.path.join(imgs_path, img_path)) for img_path in imgs_path_list]

def warp(x, f):
  B, C, H, W = x.size()
        
  xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
  yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
  xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
  yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
  grid = torch.cat((xx ,yy) ,1).float().cuda()
  
  vgrid = torch.autograd.Variable(grid) + f
  vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1) - 1.0
  vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1) - 1.0
  vgrid = vgrid.permute(0 ,2 ,3 ,1)
  
  return F.grid_sample(x, vgrid, align_corners=False)

def create_task_folders(eval_dir, task):
  path_ref = os.path.join(eval_dir, task + "/ref")
  path_fake = os.path.join(eval_dir, task + "/fake")
  
  if os.path.exists(path_ref):
    #print("exists")
    shutil.rmtree(path_ref, ignore_errors=True)
  os.makedirs(path_ref)
    
  if os.path.exists(path_fake):
    #print("exists")
    shutil.rmtree(path_fake, ignore_errors=True)
  os.makedirs(path_fake)

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, eval_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = False#config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""

        data_loader = self.train_loader
        #src_img, src_img2, src_lbl, ref_img, ref_lbl, mask, flow

        c_fixed_list = []
        
        for c in range(self.c_dim):
          lbl_org = np.zeros((self.batch_size, self.c_dim))
          lbl_org[:,c] = 1
          lbl_org = torch.tensor(lbl_org).float().to(self.device)
          c_fixed_list.append(lbl_org)
          
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, _, c_org, _, _, _, _ = next(data_iter)
        x_fixed = x_fixed.to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
          start_iters = self.resume_iters
          self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
              x_real, _, label_org, _, label_trg, _, _ = next(data_iter)
            except:
              data_iter = iter(data_loader)
              x_real, _, label_org, _, label_trg, _, _ = next(data_iter)

            # Generate target domain labels randomly.
            #rand_idx = torch.randperm(label_org.size(0))
            #label_trg = label_org[rand_idx]
            
            b_size = x_real.shape[0]

            c_org = np.zeros((b_size, self.c_dim))
            c_trg = np.zeros((b_size, self.c_dim))

            for bs in range(b_size):
              c_org[bs, label_org[bs]] = 1
              c_trg[bs, label_trg[bs]] = 1

            label_org = torch.tensor(c_org).float()
            label_trg = torch.tensor(c_trg).float()
            
            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        data_loader = self.eval_loader
        
        with torch.no_grad():
            for i, (x_real, _, c_org, _, _, _, _) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
    #==================================================================================================
    
    @torch.no_grad()
    def eval(self):
      self.restore_model(self.test_iters)
      
      print('Calculating evaluation metrics...')
      #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
      eval_dir = os.getcwd() + "/eval/"
      print(eval_dir)
      eval_loader = self.eval_loader
      
      if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
      num_domains = 4
      
      #generate_new = True
    
      #num_files = sum([len(files) for r, d, files in os.walk(args.eval_dir)])
      #print("num_files", num_files, len(eval_loader), (1 + args.num_outs_per_domain)*len(eval_loader)*args.batch_size)
    
      #if num_files != (1 + args.num_outs_per_domain)*len(eval_loader):
      #shutil.rmtree(args.eval_dir, ignore_errors=True)
      #os.makedirs(args.eval_dir)
      generate_new = True
      
      tcl_dict = {}
      # prepare
      for d in range(1, num_domains):
        src_domain = "style0"
        trg_domain = "style" + str(d)
        
        t1 = '%s2%s' % (src_domain, trg_domain)
        t2 = '%s2%s' % (trg_domain, src_domain)
        
        tcl_dict[t1] = []
        tcl_dict[t2] = []
        
        if generate_new:
          create_task_folders(eval_dir, t1)
          #create_task_folders(eval_dir, t2)

      # generate
      for i, x_src_all in enumerate(tqdm(eval_loader, total=len(eval_loader))):
        x_real, x_real2, y_org, x_ref, y_trg, mask, flow = x_src_all
        
        x_real = x_real.to(self.device)
        x_real2 = x_real2.to(self.device)
        y_org = y_org.to(self.device)
        x_ref = x_ref.to(self.device)
        y_trg = y_trg.to(self.device)
        mask = mask.to(self.device)
        flow = flow.to(self.device)
        
        N = x_real.size(0)
    
        for k in range(N):
          y_org_np = y_org[k].cpu().numpy()
          y_trg_np = y_trg[k].cpu().numpy()
          src_domain = "style" + str(y_org_np)
          trg_domain = "style" + str(y_trg_np)
          
          if src_domain == trg_domain or y_trg_np == 0:
            continue
          
          task = '%s2%s' % (src_domain, trg_domain)
    
          b_size = x_real.shape[0]
          c_trg = np.zeros((b_size, self.c_dim))

          for bs in range(b_size):
            c_trg[bs, y_trg[bs]] = 1

          c_trg = torch.tensor(c_trg).float().to(self.device)
    
          x_fake = self.G(x_real, c_trg)
          x_fake2 = self.G(x_real2, c_trg)
          
          x_warp = warp(x_fake, flow)
          tcl_err = ((mask*(x_fake2 - x_warp))**2).mean(dim=(1, 2, 3))**0.5
          
          tcl_dict[task].append(tcl_err[k].cpu().numpy())
    
          path_ref = os.path.join(eval_dir, task + "/ref")
          path_fake = os.path.join(eval_dir, task + "/fake")
    
          #if not os.path.exists(path_ref):
          #  os.makedirs(path_ref)
            
          #if not os.path.exists(path_fake):
          #  os.makedirs(path_fake)
          
          if generate_new:
            filename = os.path.join(path_ref, '%.4i.png' % (i*self.batch_size+(k+1)))
            utils.save_image(x_ref[k], ncol=1, filename=filename)
          
          filename = os.path.join(path_fake, '%.4i.png' % (i*self.batch_size+(k+1)))
          utils.save_image(x_fake[k], ncol=1, filename=filename)
    
          #filename = os.path.join(args.eval_dir, task + "/tcl_losses.txt")
          #with open(filename, "a") as text_file:
          #  text_file.write(str(tcl_err[k].cpu().numpy()) + "\n")
    
      # evaluate
      print("computing fid, lpips and tcl")
    
      tasks = [dir for dir in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, dir))]
      tasks.sort()
    
      # fid and lpips
      fid_values = OrderedDict()
      #lpips_dict = OrderedDict()
      tcl_values = OrderedDict()
      for task in tasks:
        print(task)
        path_ref = os.path.join(eval_dir, task + "/ref")
        path_fake = os.path.join(eval_dir, task + "/fake")
        #path_tcl = os.path.join(args.eval_dir, task + "/tcl_losses.txt")
      
        #fake_group = load_images(path_fake)
        
        #with open(path_tcl, "r") as text_file:
        #  tcl_data = text_file.read()
        
        #tcl_data = tcl_data.split("\n")[:-1]
        #tcl_data = [float(td) for td in tcl_data]
        tcl_data = tcl_dict[task]
          
        print("TCL", len(tcl_data))
        tcl_mean = np.array(tcl_data).mean()
        print(tcl_mean)
        tcl_values['TCL_%s' % (task)] = float(tcl_mean)
        
        '''
        lpips_values = []
        fake_chunks = chunks(fake_group, 1)
        for cidx in range(len(fake_chunks)):
          lpips_value = calculate_lpips_given_images(fake_chunks[cidx])
          lpips_values.append(lpips_value)
        
        
        print("LPIPS")
        # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
        lpips_mean = np.array(lpips_values).mean()
        lpips_dict['LPIPS_%s' % (task)] = lpips_mean
        '''
    
        print("FID")
        fid_value = calculate_fid_given_paths(paths=[path_ref, path_fake], img_size=256, batch_size=self.batch_size)
        fid_values['FID_%s' % (task)] = fid_value
      
      '''
      # calculate the average LPIPS for all tasks
      lpips_mean = 0
      for _, value in lpips_dict.items():
          lpips_mean += value / len(lpips_dict)
      lpips_dict['LPIPS_mean'] = lpips_mean
    
      # report LPIPS values
      filename = os.path.join(args.eval_dir, 'LPIPS.json')
      utils.save_json(lpips_dict, filename)'''
      
      # calculate the average FID for all tasks
      fid_mean = 0
      #fid_means = [[], [], []]
      for key, value in fid_values.items():
        #for d in range(1, num_domains):
        #  if str(d) in key:
        #    fid_means[d-1].append(value)
        fid_mean += value / len(fid_values)
      
      #for d in range(1, num_domains):
      #  fid_values['FID_s%d_mean' % d] = np.array(fid_means[d-1]).mean()
      
      fid_values['FID_mean'] = fid_mean
    
      # report FID values
      filename = os.path.join(eval_dir, 'FID.json')
      utils.save_json(fid_values, filename)
      
      # calculate the average TCL for all tasks
      tcl_mean = 0
      #tcl_means = [[], [], []]
      for _, value in tcl_values.items():
        #for d in range(1, num_domains):
        #  if str(d) in key:
        #    tcl_means[d-1].append(value)
        #print(value, len(tcl_values))
        tcl_mean += value / len(tcl_values)
      #print(tcl_mean)
      #for d in range(1, num_domains):
      #  tcl_values['TCL_s%d_mean' % d] = np.array(tcl_means[d-1]).mean()
      
      tcl_values['TCL_mean'] = float(tcl_mean)
    
      # report TCL values
      filename = os.path.join(eval_dir, 'TCL.json')
      utils.save_json(tcl_values, filename)
    
    @torch.no_grad()
    def eval_sintel(self, args, sintel_dir="G:/Datasets/MPI-Sintel-complete/"):
      self.restore_model(self.test_iters)
      out_path = os.getcwd() + "/sintel_eval/"      

      raft_model = initRaftModel(args, self.device)
      num_domains = 4
      
      transform = []
      transform.append(transforms.ToTensor())
      transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
      transform = transforms.Compose(transform)
      
      train_dir = os.path.join(sintel_dir, "training", "final")
      train_list = os.listdir(train_dir)
      train_list.sort()
      
      test_dir = os.path.join(sintel_dir, "test", "final")
      test_list = os.listdir(test_dir)
      test_list.sort()
      
      video_list = [os.path.join(train_dir, vid) for vid in train_list]
      video_list += [os.path.join(test_dir, vid) for vid in test_list]
      
      vid_list = train_list + test_list
      tcl_st_dict = {}
      tcl_lt_dict = {}
      
      tcl_st_dict = OrderedDict()
      tcl_lt_dict = OrderedDict()
      dt_dict = OrderedDict()
      
      for j, vid_dir in enumerate(video_list):
        vid = vid_list[j]
    
        sintel_dset = SingleSintelVideo(vid_dir, transform)
        loader = data.DataLoader(dataset=sintel_dset, batch_size=1, shuffle=False, num_workers=0)
        
        for y in range(1, num_domains):
          #y_trg = torch.Tensor([y])[0].type(torch.LongTensor).to(device)
          key = vid + "_s" + str(y)
          vid_path = os.path.join(out_path, key)
          if not os.path.exists(vid_path):
            os.makedirs(vid_path)
        
          tcl_st_vals = []
          tcl_lt_vals = []
          dt_vals = []
          
          for i, imgs in enumerate(tqdm(loader, total=len(loader))):
            img, img_last, img_past = imgs
            
            img = img.to(self.device)
            img_last = img_last.to(self.device)
            img_past  = img_past.to(self.device)
            
            c_trg = np.zeros((1, self.c_dim))
            c_trg[:, y] = 1
            c_trg = torch.tensor(c_trg).float().to(self.device)

            t_start = time.time()
            x_fake = self.G(img, c_trg)
            t_end = time.time()
            
            dt_vals.append((t_end - t_start)*1000)
            
            if i > 0:
              tcl_st = computeTCL(self.G, raft_model, x_fake, img, img_last, c_trg)
              tcl_st_vals.append(tcl_st.cpu().numpy())
            
            if i >= 5:
              tcl_lt = computeTCL(self.G, raft_model, x_fake, img, img_past, c_trg)
              tcl_lt_vals.append(tcl_lt.cpu().numpy())
              
            filename = os.path.join(vid_path, "frame_%04d.png" % i)
            sintel_eval.save_image(x_fake[0], ncol=1, filename=filename)
            
          tcl_st_dict["TCL-ST_" + key] = float(np.array(tcl_st_vals).mean())
          tcl_lt_dict["TCL-LT_" + key] = float(np.array(tcl_lt_vals).mean())
          dt_dict["DT_" + key] = float(np.array(dt_vals).mean())
        
      save_dict_as_json("TCL-ST", tcl_st_dict, out_path, num_domains)
      save_dict_as_json("TCL-LT", tcl_lt_dict, out_path, num_domains)
      save_dict_as_json("DT", dt_dict, out_path, num_domains)
