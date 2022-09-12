"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
import warnings
import numpy as np
import torch.nn.functional as nnf
import moreutils as uvir
from models.networks.distributions import VonMisesFisher, HypersphericalUniform


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt, drop_first=self.opt.drop_first)
            self.criterionFeat = torch.nn.L1Loss()
            self.slice_consistency_activated = opt.lambda_slice_consistency != 0
            self.activation_slice_consistency = opt.activation_slice_consistency

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                if opt.type_prior == 'N':
                    self.KLDLoss = networks.KLDLoss()
                elif opt.type_prior == 'S':
                    self.KLDLoss = networks.KLDLossVMF()
            if opt.topK_discrim:
                self.topK_discrim = {'on': True, 'topK': int(0.6*opt.batchSize), 'from_epoch': 400}
            else:
                self.topK_discrim = {'on': False, 'topK': int(0.6*opt.batchSize), 'from_epoch': 400}
            if opt.D_modality_class:
                self.D_modality_class = True
                self.one_hot_mod = {}
                for s_ind, s in enumerate(opt.sequences):
                    base = torch.zeros(len(opt.sequences))
                    base[s_ind] = 1.0
                    self.one_hot_mod[s] = base
            else:
                self.D_modality_class = False
        else:
            self.topK_discrim = {'on': False}
    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, **kwargs):
        if 'get_code' in kwargs.keys():
            get_code = kwargs['get_code']
        if 'return_D_predictions' in kwargs.keys():
            get_D_pred = kwargs['return_D_predictions']
        else:
            get_D_pred = False
        if self.topK_discrim['on']:
            if 'epoch' in kwargs.keys():
                epoch = kwargs['epoch']
            else:
                ValueError("If topK_discrim flag is ON, you need to pass on the epoch number to check"
                           "if we need to activate it. Please specify epoch = X in kwargs.")

        warnings.filterwarnings("ignore")
        input_semantics, gt_image, style_image = self.preprocess_input(data)
        if mode == 'generator':
            if get_code:
                if self.topK_discrim['on']:
                    g_loss, generated, d_acc, g_loss_nw, z = self.compute_generator_loss(
                        input_semantics, style = style_image, gt = gt_image, modalities = data['this_seq'],
                        get_code = True, epoch = epoch)
                else:
                    g_loss, generated, d_acc, g_loss_nw, z = self.compute_generator_loss(
                        input_semantics, style = style_image, gt = gt_image, modalities = data['this_seq'],
                        get_code = True)
                return g_loss, generated, d_acc, g_loss_nw, z
            else:
                if self.topK_discrim['on']:
                    g_loss, generated, d_acc, g_loss_nw = self.compute_generator_loss(
                        input_semantics, style=style_image, gt=gt_image, modalities=data['this_seq'], get_code=False,
                        epoch = epoch)
                else:
                    g_loss, generated, d_acc, g_loss_nw = self.compute_generator_loss(
                        input_semantics, style = style_image, gt = gt_image, modalities = data['this_seq'],
                        get_code = False)
                return g_loss, generated, d_acc, g_loss_nw
        elif mode == 'discriminator':
            if self.topK_discrim['on']:
                if get_D_pred:
                    d_loss, d_acc, outputs_D = self.compute_discriminator_loss(
                        input_semantics, style_image, gt_image, epoch=epoch,
                        return_predictions=True, modalities=data['this_seq'])
                    return d_loss, d_acc, outputs_D
                else:
                    d_loss, d_acc = self.compute_discriminator_loss(
                        input_semantics, style_image, gt_image, epoch = epoch,
                        modalities=data['this_seq'])
                    return d_loss, d_acc
            else:
                if get_D_pred:
                    d_loss, d_acc, outputs_D = self.compute_discriminator_loss(
                        input_semantics, style_image,  gt_image, return_predictions=True,
                    modalities=data['this_seq'])
                    return d_loss, d_acc, outputs_D
                else:
                    d_loss, d_acc = self.compute_discriminator_loss(
                        input_semantics, style_image,  gt_image,
                    modalities=data['this_seq'])
                    return d_loss, d_acc
        elif mode == 'encode_only':
            if self.opt.type_prior == 'N':
                z, mu, logvar = self.encode_z(style_image)
            elif self.opt.type_prior == 'S':
                z, mu, logvar, _, _ = self.encode_z(style_image)
            return z, mu
        elif mode == 'encode_only_all':
            if self.opt.type_prior == 'N':
                z, mu, logvar, noise = self.encode_all(style_image)
                return z, mu, logvar, noise
            elif self.opt.type_prior == 'S':
                z, mu, logvar = self.encode_all(style_image)
                return z, mu, logvar
        elif mode == 'generate_mu':
            return self.generate_mu(input_semantics, style_image)
        elif mode == 'inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input_semantics, style_image)
                return fake_image
        elif mode == 'random_style_inference':
            with torch.no_grad():
                fake_image = self.generate_fake(input_semantics, style_image, random_style = True)
                return fake_image
        elif mode == 'validation':
            # Compute generator losses and forward pass validation data using torch.no_grad.
            if get_code:
                loss, generated, d_acc, loss_nw, z = self.compute_generator_loss(
                    input_semantics, style = style_image, gt = gt_image, modalities = data['this_seq'], get_code = True,
                    epoch = epoch, validation=True)
            else:
                loss, generated, d_acc, loss_nw = self.compute_generator_loss(
                    input_semantics, style = style_image, gt = gt_image, modalities = data['this_seq'], get_code = False,
                    validation=True, epoch = epoch)

            # Add discrimination loss
            with torch.no_grad():
                if self.D_modality_class:
                    input_mods = data['this_seq']
                else:
                    input_mods = None
                pred_fake, pred_real, D_acc = self.discriminate(input_semantics, generated, gt_image,
                                                                input_mods)
                loss['D_Fake'] = loss_nw['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
                loss['D_real'] = loss_nw['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
                loss['D_acc_fakes'] = loss_nw['D_acc_fakes'] = D_acc['D_acc_fakes']
                loss['D_acc_reals'] = loss_nw['D_acc_reals'] = D_acc['D_acc_reals']
                loss['D_acc_total'] = loss_nw['D_acc_total'] = D_acc['D_acc_total']

            # Uncuda cuda elements
            for loss_item, loss_val in loss.items():
                if loss_val.is_cuda:
                    loss[loss_item] = loss_val.detach().cpu()
                loss[loss_item] = loss_val.item()
                if loss_nw[loss_item].is_cuda:
                    loss_nw[loss_item] =  loss_nw[loss_item].detach().cpu()
                loss_nw[loss_item] = loss_nw[loss_item].item()
            # Return
            if get_code:
                return loss, generated, d_acc, loss_nw, z
            else:
                return loss, generated, d_acc, loss_nw

        elif mode == 'inference_code':
            with torch.no_grad():
                fake_image, code = self.generate_fake_code(input_semantics, style_image)
                return fake_image, code
        elif mode == 'generator_test':
            # We generate an image without computing losses like in gmode.
            with torch.no_grad():
                encoded = self.generate_noLosses(input_semantics, style_image)
                return encoded
        elif mode == 'generator_no_losses':
            # We generate an image without computing losses like in gmode.
            encoded = self.generate_noLosses(input_semantics, style_image)
            return encoded
        elif mode == 'encode_tweak_decode':
            if 'n_codes' in kwargs.keys():
                n_codes = kwargs['n_codes']
            else:
                n_codes = 1
            ### Encoding, tweaking and decoding
            return self.encode_tweak_decode(input_semantics, style_image, n_codes)
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt, parameters_from_mod_disc = None):
        G_params = list(self.netG.parameters())
        if opt.mod_disc and parameters_from_mod_disc is not None:
            G_params += parameters_from_mod_disc
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / opt.TTUR_factor, opt.lr * opt.TTUR_factor

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G',  epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):

        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        if opt.isTrain:
            if opt.pretrained_E is not None and not opt.continue_train:
                netE = util.load_network_from_file(netE, opt.pretrained_E)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        #data['label'] = data['label'].long()
        #data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['style_image'] = data['style_image'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc # Adds 1 if there is a "dontcare" label
        #input_label = self.FloatTensor(bs, nc, h, w).zero_()
        #input_semantics = input_label.scatter_(1, label_map, 1.0)
        input_semantics = label_map

        return input_semantics, data['image'], data['style_image']

    def compute_generator_loss(self, input_semantics, style, gt, modalities, get_code, validation = False,
                               epoch = None):

        # If real_gt is None, we calculate the losses with regards to image.
        # Otherwise, with regards to real_gt (loss free).

        G_losses = {}
        G_losses_nw = {}

        if get_code:
            fake_image, z, KLD_loss = self.generate_fake(
                    input_semantics, style, compute_kld_loss=self.opt.use_vae, modalities=modalities,
            get_code = True)
        else:
            fake_image, KLD_loss = self.generate_fake(
                    input_semantics, style, compute_kld_loss=self.opt.use_vae, modalities=modalities,
            get_code = False)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
            G_losses_nw['KLD'] = KLD_loss / self.opt.lambda_kld

        if self.D_modality_class:
            modalities_to_disc = modalities
        else:
            modalities_to_disc = None

        if self.slice_consistency_activated and epoch >= self.activation_slice_consistency:
            fake_image_bis = self.generate_fake(input_semantics, style, compute_kld_loss=False, modalities=modalities,
                                            get_code=False)
            G_losses_nw['slice-con'] = self.criterionFeat(fake_image, fake_image_bis)
            G_losses['slice-con'] = self.opt.lambda_slice_consistency * G_losses_nw['slice-con']


        if validation:
            with torch.no_grad():
                pred_fake, pred_real, accs = self.discriminate(
                    input_semantics, fake_image, gt, modalities_to_disc)
                G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                        for_discriminator=False)
                G_losses_nw['GAN'] = G_losses['GAN']
        else:
            pred_fake, pred_real, accs = self.discriminate(
                input_semantics, fake_image, gt, modalities_to_disc)
            if self.topK_discrim['on'] and epoch > self.topK_discrim['from_epoch']:
                acc = [torch.mean(torch.sigmoid(i[-1]).view(i[-1].shape[0], -1), -1) for i in pred_fake]
                acc = torch.stack(acc, 0)
                acc = torch.mean(acc, 0)
                pred_fake_GAN = []
                for j in pred_fake:
                    pred_fake_GAN.append([])
                    for i in j:
                        topk_indices = torch.topk(acc, self.topK_discrim['topK'], dim=0)[-1]
                        pred_fake_GAN[-1].append(i[torch.topk(acc, self.topK_discrim['topK'], dim=0)[-1], ...])
                G_losses['GAN'] = self.criterionGAN(pred_fake_GAN, True, for_discriminator=False)
            else:
                G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                    for_discriminator=False)
            G_losses_nw['GAN'] = G_losses['GAN']

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            GAN_Feat_loss_nw = 0
            if validation:
                with  torch.no_grad():
                    for i in range(num_D):  # for each discriminator
                        # last output is the final prediction, so we exclude it
                        num_intermediate_outputs = len(pred_fake[i]) - 1
                        for j in range(num_intermediate_outputs):  # for each layer output
                            unweighted_loss = self.criterionFeat(
                                pred_fake[i][j], pred_real[i][j].detach())
                            GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                            GAN_Feat_loss_nw += (unweighted_loss / num_D)
                    G_losses['GAN_Feat'] = GAN_Feat_loss
                    G_losses_nw['GAN_Feat'] = GAN_Feat_loss_nw
            else:
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                        GAN_Feat_loss_nw += (unweighted_loss / num_D).item()
                G_losses['GAN_Feat'] = GAN_Feat_loss
                G_losses_nw['GAN_Feat'] = GAN_Feat_loss_nw

        if not self.opt.no_vgg_loss:
            if validation:
                with torch.no_grad():
                    G_losses['VGG'] = self.criterionVGG(fake_image, gt) \
                                      * self.opt.lambda_vgg
                    G_losses_nw['VGG'] = G_losses['VGG'] / self.opt.lambda_vgg
            else:
                G_losses['VGG'] = self.criterionVGG(fake_image, gt) \
                    * self.opt.lambda_vgg
                G_losses_nw['VGG'] = G_losses['VGG'] / self.opt.lambda_vgg

        for loss, loss_val in G_losses.items():
            if torch.isnan(loss_val):
                ValueError("Nan found in %s" %loss)
        return G_losses, fake_image, accs, G_losses_nw, z

    def generate_noLosses(self, input_semantics, img):
        # Custom Code. Returns the decoded encoded version of an image.
        z = None
        if self.opt.use_vae:
            #mu, _ = self.netE(real_image)
            #z = mu
            if self.opt.type_prior == 'N':
                z, _, _ = self.encode_z(img)
            elif self.opt.type_prior == 'S':
                z, _, _, _, _ = self.encode_z(img)
        encoded = self.netG(input_semantics, z=z)

        return encoded

    def compute_discriminator_loss(self, input_semantics, style, gt, epoch = None,
                                   return_predictions = False, modalities = None):
        if self.D_modality_class:
            if modalities is None:
                ValueError("D_modality_class is active. You need to input the modalities to "
                           "the discriminator.")
            else:
                modalities_to_disc = modalities
        else:
            modalities_to_disc = None

        D_losses = {}

        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, style, get_code = False)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        # Ground truth: real images, can be noisy (comes from the 'image')
        pred_fake, pred_real, D_acc = self.discriminate(input_semantics, fake_image, gt,
                                                        modalities_to_disc)

        if self.topK_discrim['on'] and epoch > self.topK_discrim['from_epoch']:
            acc = [torch.mean(torch.sigmoid(i[-1]).view(i[-1].shape[0], -1), -1) for i in pred_fake]
            acc = torch.stack(acc, 0)
            acc = torch.mean(acc, 0)
            pred_fake_GAN = []
            pred_real_GAN = []
            for j_ind, j in enumerate(pred_fake):
                pred_fake_GAN.append([])
                pred_real_GAN.append([])
                for i_ind, i in enumerate(j):
                    topk_indices = torch.topk(acc, self.topK_discrim['topK'], dim=0)[-1]
                    pred_fake_GAN[-1].append(i[torch.topk(acc, self.topK_discrim['topK'], dim=0)[-1], ...])
                    pred_real_GAN[-1].append(pred_real[j_ind][i_ind][torch.topk(acc, self.topK_discrim['topK'], dim=0)[-1], ...])


            D_losses['D_real'] = self.criterionGAN(pred_real_GAN, True, for_discriminator=True,
                                                   )
            D_losses['D_Fake'] = self.criterionGAN(pred_fake_GAN, False, for_discriminator=True)
        else:
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)

        if return_predictions:
            outputs_D = {'real': gt.detach().cpu(), 'fake': fake_image.detach().cpu(),
                         'pred_real': [], 'pred_fake': []}
            for ind, i in enumerate(pred_fake):
                if isinstance(i, list):
                    outputs_D['pred_fake'].append(i[-1])
                    outputs_D['pred_real'].append(pred_real[ind][-1])
                else:
                    outputs_D['pred_fake'].append(i)
                    outputs_D['pred_real'].append(pred_real[ind])

            for loss, loss_val in D_losses.items():
                if torch.isnan(loss_val):
                    ValueError("Nan found in %s" % loss)
            return D_losses, D_acc, outputs_D
        else:
            for loss, loss_val in D_losses.items():
                if torch.isnan(loss_val):
                    ValueError("Nan found in %s" % loss)
            return D_losses, D_acc

    def encode_z(self, img):
        mu, logvar = self.netE(img)
        if self.opt.type_prior == 'N':
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        elif self.opt.type_prior == 'S':
            z, q_z, p_z = self.reparameterize(mu, logvar)
            return z, mu, logvar, q_z, p_z

    def encode_all(self, img):
        mu, logvar = self.netE(img)
        if self.opt.type_prior=='N':
            z, noise = self.reparameterize_wnoise(mu, logvar)
            return z, mu, logvar, noise
        else:
            z, _, _ = self.reparameterize_wnoise(mu, logvar)
            return z, mu, logvar


    def encode_tweak_decode(self, input_semantics, img, n_codes):
        '''
        Encodes a batch of images, then fetches the N_CODES biggest indices of
        the latent code, and zeroes them turn by turn, generating the images.
        :param input_semantics: Semantic map
        :param real_image: Image
        :param n_codes: Number of TOPK items of the code (code dimensions) to take into account
        :return: output: CODE_IND x B x C x H x W images where CODE_IND is the number of different codes,
        and indices_used: CODE_IND x B, indices removed from the respective images.
        '''

        # Encode
        z = None
        if self.opt.use_vae:
            mu, _ = self.netE(img)
            z = mu

        # We take the biggest: means of difference dimensions
        _, top_indices = torch.topk(z, n_codes, dim = 1)
        output = []
        indices_used = torch.zeros(n_codes, z.shape[0])
        for i in range(top_indices.shape[1]):
            ind_ = top_indices[:, i]
            z_ = z
            for b in range(z_.shape[0]):
                z_[:, ind_[b]] = 0
                indices_used[i, b] = ind_[b]
            output.append(self.netG(input_semantics, z = z_).detach().cpu().unsqueeze(0))

        return torch.cat(output), indices_used

    def generate_mu(self, input_semantics, img):
        '''
        Encodes a batch of images, then fetches the N_CODES biggest indices of
        the latent code, and zeroes them turn by turn, generating the images.
        :param input_semantics: Semantic map
        :param real_image: Image
        :return: output: Generated images from the mean of the distribution of the style encoded.
        '''

        # Encode
        z = None
        if self.opt.use_vae:
            mu, _ = self.netE(img)
            z = mu

        return self.netG(input_semantics, z = z).detach().cpu()


    def generate_fake(self, input_semantics, style_image, compute_kld_loss=False,
                      modalities = None, get_code = False, random_style = False):
        z = None
        KLD_loss = None
        if self.opt.use_vae and not random_style:
            if self.opt.type_prior == 'N':
                z, mu, logvar = self.encode_z(style_image)
                if compute_kld_loss:
                    KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
            elif self.opt.type_prior == 'S':
                z, mu, logvar, q_z, p_z = self.encode_z(style_image)
                if compute_kld_loss:
                    KLD_loss = self.KLDLoss(q_z, p_z) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        if get_code:
            if compute_kld_loss:
                return fake_image, z, KLD_loss
            else:
                return  fake_image, z
        else:
            if compute_kld_loss:
                return fake_image, KLD_loss
            else:
                return fake_image

    def generate_fake_code(self, input_semantics, img):
        '''
        Generates a synthetic image and returns it, as well as the VAE code.
        :param input_semantics:
        :param real_image:
        :param compute_kld_loss:
        :return:
        :return:
        '''
        z = None
        if self.opt.use_vae:
            if self.opt.type_prior == 'N':
                z, mu, logvar = self.encode_z(img)
            elif self.opt.type_prior == 'S':
                z, _, _, _, _ = self.encode_z(img)
        fake_image = self.netG(input_semantics, z=z)
        return fake_image, z

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, modalities = None):
        if self.D_modality_class and modalities is None:
            ValueError("You need to pass modalities as parameters if D_modality_class is activated. "
                       "The descriminator needs to have the modalities passed as input.")

        input_semantics = uvir.oneHotPVM(input_semantics)

        if self.D_modality_class:
            # We create a BxNUM_MODxHxW vector.
            i_m = self.create_oneHotMOD(modalities)
            ns = [1, 1] + list(input_semantics.shape[2:])
            for s_ind in input_semantics.shape[2:]:
                i_m = i_m.unsqueeze(-1)
            i_m = i_m.repeat(ns).type(fake_image.type())
            fake_concat = torch.cat([input_semantics, i_m, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, i_m, real_image], dim=1)
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        accs = self.accuracy_Discrim(discriminator_out, semantic = input_semantics, drop_first = self.opt.drop_first)
        #accs = self.yy_Discrim(discriminator_out) # We don't pass semantics here

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real, accs

    def create_oneHotMOD(self, modalities):

        mod_vector = []
        for m in modalities:
            mod_vector.append(self.one_hot_mod[m])
        mod_vector = torch.stack(mod_vector)

        return mod_vector

    # Take the prediction of fake and real images from the combined batch

    def accuracy_Discrim(self, pred, **kwargs):
        '''
        :param pred:
        :return:
        '''

        if 'semantic' in kwargs.keys():
            semantic_map = kwargs['semantic']
            # We bring to 1 channel, disregarding the background.
            semantic_map = semantic_map[:, 1:,...].sum(dim = 1).unsqueeze(1)

        else:
            semantic_map = None
        # We need to get the output of each of the discriminators

        if 'drop_first' in kwargs.keys():
            drop_first = kwargs['drop_first']
        else:
            drop_first = False

        total_accs = {'D_acc_fakes': [], 'D_acc_reals': [], 'D_acc_total': []}

        for ind, D in enumerate(pred):

            if (drop_first and ind!=0) or not drop_first:

                pred_ = D[-1] # Last item is discriminator output
                pred_s = torch.sigmoid(pred_) #SMOOTH ACCURACY
                #pred_s = torch.sigmoid(pred_).round()
                n_ims = int(np.round(pred_s.shape[0] / 2))
                ground_truth_real = torch.ones([n_ims, ] + list(pred_s.shape[1:])).cuda()
                ground_truth_fake = torch.zeros([n_ims, ] + list(pred_s.shape[1:])).cuda()
                pred_s_fake = pred_s[:n_ims, ...]
                pred_s_real = pred_s[n_ims:, ...]

                if semantic_map is not None:
                    semantic_map_rs = nnf.interpolate(semantic_map,
                                                      size=(pred_s_fake.shape[-2],
                                                            pred_s_fake.shape[-1]),
                                                      mode='bicubic', align_corners=False)
                    semantic_map_rs = semantic_map_rs.round().float() != 0.0
                    # pos_vals_fakes = ground_truth_fake == pred_s_fake
                    # pos_vals_reals = ground_truth_real == pred_s_real
                    pos_vals_fakes = 1.0 - pred_s_fake
                    pos_vals_reals = pred_s_real
                    if True not in semantic_map_rs:
                        # Empty
                        total_accs['D_acc_fakes'].append(torch.tensor(0.5).float())
                        total_accs['D_acc_reals'].append(torch.tensor(0.5).float())
                    else:
                        total_accs['D_acc_fakes'].append(
                            torch.mean((pos_vals_fakes[semantic_map_rs]).float())
                        )
                        total_accs['D_acc_reals'].append(
                            torch.mean((pos_vals_reals[semantic_map_rs]).float())
                        )

                else:
                    # total_accs['D_acc_fakes'].append(
                    #     torch.mean((ground_truth_fake == pred_s_fake).float()))
                    # total_accs['D_acc_reals'].append(
                    #     torch.mean((ground_truth_real == pred_s_real).float()))
                    total_accs['D_acc_fakes'].append(
                        torch.mean((ground_truth_fake - pred_s_fake).float()))
                    total_accs['D_acc_reals'].append(
                        torch.mean((pred_s_real).float()))

        total_accs['D_acc_fakes'] = torch.tensor(total_accs['D_acc_fakes']).mean()
        total_accs['D_acc_reals'] = torch.tensor(total_accs['D_acc_reals']).mean()
        total_accs['D_acc_total'] = (total_accs['D_acc_fakes'] + total_accs['D_acc_reals']) / 2

        # We detach
        ground_truth_fake = ground_truth_fake.detach().cpu()
        ground_truth_real = ground_truth_real.detach().cpu()

        return total_accs

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):

        if self.opt.type_prior == 'N':
            # Normal distribution.
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
        elif self.opt.type_prior == 'S':
            q_z = VonMisesFisher(mu, logvar)
            p_z = HypersphericalUniform(self.opt.z_dim, -1, device=q_z.device)
            z = q_z.rsample()
            return z, q_z, p_z

    def reparameterize_wnoise(self, mu, logvar):
        if self.opt.type_prior == 'N':
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu, eps
        elif self.opt.type_prior == 'S':
            q_z = VonMisesFisher(mu, logvar)
            p_z = HypersphericalUniform(self.opt.z_dim, -1)
            z = q_z.rsample()
            return z, q_z, p_z


    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def freezeDecoder(self):
        '''
        Freezes the weights of the decoder, making them non-trainable.
        :return:
        '''
        for param in self.netG.parameters():
            param.requires_grad = False

    def unfreezeDecoder(self):
        '''
        Freezes the weights of the encoder, making them non-trainable.
        :return:
        '''
        for param in self.netG.parameters():
            param.requires_grad = True

    def getLastGenLayer(self):
        return None

