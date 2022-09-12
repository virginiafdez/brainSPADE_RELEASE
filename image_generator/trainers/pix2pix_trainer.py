"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# test
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
import torch
import util.util as util
import numpy as np
from models.modality_discrimination.modisc_v2 import Modisc
from models.networks.loss import coSimI, l1_norm, l2_norm
import shutil
import monai
import os

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        self.g_losses = None
        self.g_losses_noweight = None
        self.d_losses = None
        self.d_accuracy = None
        if opt.isTrain:
            self.old_lr = opt.lr
            self.disc_threshold = {'low': opt.disc_acc_lowerth, 'up': opt.disc_acc_upperth}
            # Modality discriminator added
            if opt.mod_disc:
                if opt.mod_disc_path is not None and not os.path.isfile(os.path.join(opt.checkpoints_dir, opt.name, "%d_net_MD.pth" %opt.mod_disc_epoch)):
                    shutil.copy2(opt.mod_disc_path, os.path.join(opt.checkpoints_dir, opt.name, "%d_net_MD.pth" %opt.mod_disc_epoch))

                self.modalities = self.opt.sequences
                self.datasets = self.opt.datasets
                self.mod_disc = Modisc(len(self.modalities), len(self.datasets), 0.2, 2, 1).cuda()
                # ERASE IF WORKS
                # self.mod_disc = densenet.densenet121(spatial_dims=2, in_channels=1, out_channels=len(opt.sequences),
                #                                      dropout_prob=0.2).cuda()
                if opt.mod_disc_epoch is None:
                    takeover_mod_disc = -1
                else:
                    takeover_mod_disc = opt.mod_disc_epoch
                self.mod_disc = util.load_network(self.mod_disc, 'MD', takeover_mod_disc, opt, strict = False)

                # Freeze all but last layers
                self.mod_disc_trainable = opt.train_modisc
                mod_disc_params = self.freezeModiscLayers()

                # If pre-trained and never trained
                self.mod_disc_criterion = torch.nn.BCEWithLogitsLoss()
                self.alpha_mod_disc = {'modality': self.opt.lambda_mdmod, 'dataset': self.opt.lambda_mddat}
                self.targets_mod = {} # Targets for modality classification
                self.targets_dat = {} # Targets for dataset classification
                for mod_ind, mod in enumerate(opt.sequences):
                    self.targets_mod[mod] = torch.zeros(len(opt.sequences))
                    self.targets_mod[mod][mod_ind] = 1.0
                for dat_ind, dat in enumerate(opt.datasets):
                    self.targets_dat[dat] = torch.zeros(len(opt.datasets))
                    self.targets_dat[dat][dat_ind] = 1.0

            else:
                self.mod_disc = None
                mod_disc_params = None

            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt, parameters_from_mod_disc=mod_disc_params)
            if self.opt.continue_train:
                self.loadOptimizers()
            # Self supervised loss for encoder
            if opt.self_supervised_training != 0:
                self.self_supervised = True
                self.augmentations =  monai.transforms.Compose(
                    [monai.transforms.RandAffine(prob = 1.0, scale_range=(0.2, 0.2, 0.2),
                                                 rotate_range=(0.75, 0.75, 0.75),
                                                 translate_range=(3, 3, 3),
                                                 mode="bilinear",
                                                 as_tensor_output=False,
                                                 padding_mode='border'),
                     ])

                self.ss_w = opt.self_supervised_training
                if opt.distance_metric == 'l1':
                    self.distance = l1_norm
                elif opt.distance_metric == 'l2':
                    self.distance = l2_norm
                elif opt.distance_metric == 'cosim':
                    self.distance = coSimI
                else:
                    ValueError("Distance metric can only be l1, l2 or cosim (cosine similarity)")
            else:
                self.self_supervised = False
        self.batch_accumulation = {}
        self.batch_accumulation['curr_D'] = 0
        self.batch_accumulation['curr_G'] = 0
        if opt.batch_acc_fr > 0:
            self.batch_accumulation['on'] = True
            self.batch_accumulation['freq'] = opt.batch_acc_fr
        else:
            self.batch_accumulation['on'] = False
            self.batch_accumulation['freq'] = 0

        self.nulldec = opt.nullDecoderLosses

        self.freezeE = False
        if self.opt.pretrained_E is not None:
            if self.opt.freezeE:
                self.freezeE = True

        self.gradient_norm_modisc = []
        self.last_hooks = {}
        if opt.activations_freq is not None:
            self.registerHooks()
        if opt.topK_discrim:
            self.topK_discrim = True
        else:
            self.topK_discrim = False


    def computeModalityLoss(self, data, generated, validation = False):
        '''
        Passes the generated images through a modality-classification network.
        :param data: Input data. Only the ground truth modalities (data['this_seq']) are used
        :param generated: Batch of images generated by the network
        :return: Returns the BCE Loss of the classified modalities VS ground truth modalities.
        '''

        target_mod = torch.zeros(len(data['this_seq']), len(self.modalities))# No one hot target
        for b in range(len(data['this_seq'])):
            target_mod[b, :] = self.targets_mod[data['this_seq'][b]]
        # Ground truth for dataset: we take that of the style image, not that of the ground truth image.
        target_dat = torch.zeros(len(data['this_dataset']), len(self.datasets))# No one hot target
        for b in range(len(data['this_dataset'])):
            target_dat[b, :] = self.targets_dat[data['this_dataset'][b]]

        if validation:
            with torch.no_grad():
                target_mod = target_mod.type(generated.dtype).cuda()
                target_dat = target_dat.type(generated.dtype).cuda()
                logits_mod, logits_dat = self.mod_disc(generated)
                logits_mod_gt, logits_dat_gt = self.mod_disc(data['image'].cuda())
                loss_mod = (self.mod_disc_criterion(logits_mod, target_mod)+
                                                            self.mod_disc_criterion(logits_mod_gt, target_mod)).mean()
                loss_dat = (self.mod_disc_criterion(logits_dat, target_dat)+
                                                           self.mod_disc_criterion(logits_dat_gt, target_dat))
        else:
            target_mod = target_mod.type(generated.dtype).cuda()
            target_dat = target_dat.type(generated.dtype).cuda()
            logits_mod, logits_dat = self.mod_disc(generated)
            logits_mod_gt, logits_dat_gt = self.mod_disc(data['image'].cuda())
            loss_mod = (self.mod_disc_criterion(logits_mod, target_mod) +
                        self.mod_disc_criterion(logits_mod_gt, target_mod)).mean()
            loss_dat = (self.mod_disc_criterion(logits_dat, target_dat) +
                        self.mod_disc_criterion(logits_dat_gt, target_dat))
        return loss_mod, loss_dat

    def computeModalityAccuracy(self, data, generated):
        '''
        Passes the generated images through a modality-classification network. Returns accuracy.
        :param data: Input data. Only the ground truth modalities (data['this_seq']) are used
        :param generated: Batch of images generated by the network
        :return: Returns the accuracy.
        '''

        # We build up the target tensor.
        target_mod = torch.zeros(len(data['this_seq']), len(self.modalities))# No one hot target
        for b in range(len(data['this_seq'])):
            target_mod[b, :] = self.targets_mod[data['this_seq'][b]]

        # Ground truth for dataset: we take that of the style image, not that of the ground truth image.
        target_dat = torch.zeros(len(data['this_dataset']), len(self.datasets))# No one hot target
        for b in range(len(data['this_dataset'])):
            target_dat[b, :] = self.targets_dat[data['this_dataset'][b]]

        # Load target
        target_mod = target_mod.type(generated.dtype)
        target_dat = target_dat.type(generated.dtype)

        # Forward pass it
        logits_mod, logits_dat = self.mod_disc(generated)
        #logits_mod_gt, logits_dat_gt = self.mod_disc(data['image'][:, 1:2, ...].cuda()) # GT as well

        # Accuracies
        acc_mod = (np.argmax(torch.softmax(logits_mod.detach().cpu(), 1), 1)
                   == np.argmax(target_mod, 1)).float().mean() * 100.0
        acc_dat = (np.argmax(torch.softmax(logits_dat.detach().cpu(), 1), 1)
                   == np.argmax(target_dat, 1)).float().mean() * 100.0

        return acc_mod, acc_dat

    def checkDiscriminatorLosses(self, threshold = 1.0):
        '''
        Returns true if the cross entropy loss of the modality discriminator is greater than one
        :return:
        '''

        if self.mod_disc is not None and self.g_losses is not None:
            if self.g_losses['mod_disc'] > threshold:
                return True
        else:
            return False

    def run_generator_one_step(self, data, with_gradients = False, epoch = None):

        if self.topK_discrim and epoch is None:
            ValueError("Epoch must be specified if self.topK_discrim is True.")

        self.pix2pix_model.module.unfreezeDecoder()
        if self.freezeE:
            self.pix2pix_model.module.freeze_unfreeze_Encoder(True)

        g_losses, generated, accs, g_losses_nw, z = self.pix2pix_model(data, mode='generator', get_code = True,
                                                                        epoch = epoch)

        if self.mod_disc is not None:
            loss_mod, loss_dat = self.computeModalityLoss(data, generated)
            g_losses['mod_disc'] = self.alpha_mod_disc['modality']*loss_mod
            g_losses['dat_disc'] = self.alpha_mod_disc['dataset']*loss_dat
            g_losses_nw['mod_disc'] = loss_mod.item()
            g_losses_nw['dat_disc'] = loss_dat.item()
        self.generated = generated

        # Self-supervised encoder loss
        if self.self_supervised:
            ss_loss = self.self_supervised_lr(data, z)
            g_losses['self_supervised'] = ss_loss * self.ss_w
            g_losses_nw['self_supervised'] = ss_loss

        if with_gradients:
            # If with gradients is active, we calculate the gradients of the losses specified in
            # pairs with regards to the last layers of the modality discriminator, Generator and
            # discriminator.
            pairs = {'decoder': ['KLD', 'GAN', 'GAN_Feat', 'VGG', 'mod_disc', 'dat_disc', 'self_supervised', 'slice-con'],
                     'encoder': ['KLD', 'GAN', 'GAN_Feat', 'VGG', 'mod_disc', 'dat_disc', 'self_supervised', 'slice-con'],
                     'modisc': ['mod_disc', 'dat_disc']}

            layers = {'decoder': [list(self.pix2pix_model_on_one_gpu.netG.parameters())[-2]],
                      'encoder': [list(self.pix2pix_model_on_one_gpu.netE.parameters())[-2],
                                  list(self.pix2pix_model_on_one_gpu.netE.parameters())[-4]
                                  ]
                      }

            gradients = {}

            if self.mod_disc is not None:
                layers['modisc']= [list(self.mod_disc.parameters())[-2], list(self.mod_disc.parameters())[-4]
                                  ]

            for name_loss, val in g_losses.items():
                val.backward(retain_graph = True) # Backward the specific loss
                for structure, layer_params in layers.items():
                    if name_loss in pairs[structure]:
                        try:
                            gradients[
                                "%s_%s" %(name_loss, structure)] = torch.autograd.grad(
                                val, layer_params, retain_graph=True,create_graph=True, allow_unused=True)
                        except:
                            pass
        else:
            g_loss = sum(g_losses.values()).mean()
            g_loss.backward()

        # Print gradients in the mod_disc layers
        if self.mod_disc is not None:
            for i in self.mod_disc.parameters():
                if i.requires_grad:
                    if len(self.gradient_norm_modisc) > self.opt.print_grad_freq:
                        self.gradient_norm_modisc = self.gradient_norm_modisc[1:] + [i.grad.data.norm().item()]
                    else:
                        self.gradient_norm_modisc.append(i.grad.data.norm().item())
            if self.mod_disc_trainable:
                torch.nn.utils.clip_grad_norm(self.mod_disc.parameters(), 1.0)

        if self.batch_accumulation['on']:
            # Batch Accumulation case
            if self.batch_accumulation['curr_G'] == self.batch_accumulation['freq']:
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.batch_accumulation['curr_G'] = 0
                torch.cuda.empty_cache()
            else:
                self.batch_accumulation['curr_G'] += 1
        else:
            # Not batch accumulation case
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            torch.cuda.empty_cache()

        self.optimizer_G.step()
        self.g_losses = g_losses
        self.g_losses_noweight = g_losses_nw
        self.d_accuracy = accs

        if with_gradients:
            return gradients

    def run_encoder_one_step(self, data, dataset = None):

        # Freeze the decoder weights
        # Runs a forward pass in encode mode
        # All losses and update
        self.pix2pix_model.module.freezeDecoder()
        if self.freezeE:
            self.pix2pix_model.module.freeze_unfreeze_Encoder(True)

        g_losses, generated, accs, g_losses_nw, z = self.pix2pix_model(data, mode='generator', get_code = True)

        if self.mod_disc is not None:
            loss_mod, loss_dat = self.computeModalityLoss(data, generated)
            g_losses['mod_disc'] = self.alpha_mod_disc['modality'] * loss_mod
            g_losses['dat_disc'] = self.alpha_mod_disc['dataset'] * loss_dat
            g_losses_nw['mod_disc'] = loss_mod.item()
            g_losses_nw['dat_disc'] = loss_dat.item()
        self.generated = generated # This needs to go before NMI Loss!!!
        # Self-supervised encoder loss
        if self.self_supervised:
            ss_loss = self.self_supervised_lr(data, z)
            g_losses['self_supervised'] = ss_loss * self.ss_w
            g_losses_nw['self_supervised'] = ss_loss

        if self.nulldec:
            g_loss = g_losses['KLD']
            if self.mod_disc is not None:
                g_loss += g_losses['mod_disc']+g_losses['dat_disc']
        else:
            g_loss = sum(g_losses.values()).mean()
        self.d_accuracy = accs
        g_loss.backward()
        if self.batch_accumulation['on']:
            # Batch Accumulation case
            if self.batch_accumulation['curr_G'] == self.batch_accumulation['freq']:
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.batch_accumulation['curr_G'] = 0
                torch.cuda.empty_cache()
            else:
                self.batch_accumulation['curr_G'] += 1
        else:
            # Not batch accumulation case
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            torch.cuda.empty_cache()

        self.g_losses = g_losses

    def run_discriminator_one_step(self, data, with_gradients = False, with_activations = False,
                                   epoch = None, return_predictions = False):

        if self.topK_discrim and epoch is None:
            ValueError("Epoch must be specified if self.topK_discrim is True.")
        if self.topK_discrim:
            if return_predictions:
                d_losses, d_acc, outputs_D = self.pix2pix_model(data, mode='discriminator', epoch = epoch,
                                                                           return_D_predictions = return_predictions)
            else:
                d_losses, d_acc = self.pix2pix_model(data, mode='discriminator', epoch = epoch,
                                                     return_D_predictions = return_predictions)
        else:
            if return_predictions:
                d_losses, d_acc, outputs_D = self.pix2pix_model(data, mode='discriminator',
                                                 return_D_predictions = return_predictions)
            else:
                d_losses, d_acc = self.pix2pix_model(data, mode='discriminator',
                                                     return_D_predictions = return_predictions)

        # If with gradients is active, we calculate the gradients of the losses specified in
        # pairs with regards to the last layers of the discriminator.

        if with_gradients:

            pairs = {}
            layers = {}
            D_params = list(self.pix2pix_model_on_one_gpu.netD.parameters())
            n_params_per_D = int(len(D_params)/self.opt.num_D)
            for d in range(self.opt.num_D):
                pairs['discriminator_%d' %d] =  ['D_Fake', 'D_real']
                layers['discriminator_%d' %d] = [D_params[n_params_per_D * (d+1) - 2]]

            gradients = {}

            for name_loss, val in d_losses.items():
                val.backward(retain_graph=True)  # Backward the specific loss
                for structure, layer_params in layers.items():
                    if name_loss in pairs[structure]:
                        try:
                            gradients[
                                "%s_%s" % (name_loss, structure)] = torch.autograd.grad(
                                val, layer_params, retain_graph=True, create_graph=True, allow_unused=True)
                        except:



                            pass
        else:
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()

        if self.batch_accumulation['on']:
            # Batch Accumulation case
            if self.batch_accumulation['curr_D'] == self.batch_accumulation['freq']:
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()
                self.batch_accumulation['curr_D'] = 0
                torch.cuda.empty_cache()
            else:
                self.batch_accumulation['curr_D'] += 1
        else:
            # Not batch accumulation case
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            torch.cuda.empty_cache()

        self.d_losses = d_losses
        self.d_accuracy = d_acc

        if with_gradients:
            if return_predictions:
                return gradients, outputs_D
            else:
                return gradients
        else:
            if return_predictions:
                return outputs_D

    def self_supervised_lr(self, data, codes):

        '''
        Calclulates self-supervised learning losses by applying augmentation on the style image, and
        making sure that the distance between its code and the code of the original one are close.
        :param data:
        :param codes:
        :return: Self-supervised loss: distance metric (l2, l1 or cosine similarity, as per self.opt.distance_metric)
        '''

        data_aug = self.augmentations(data['style_image'])
        if isinstance(data_aug, monai.data.MetaTensor):
            data_aug = torch.from_numpy(data_aug.array)
        data_aug_dict = data.copy()
        data_aug_dict['style_image'] = data_aug
        codes_aug, _ = self.pix2pix_model_on_one_gpu(data_aug_dict, 'encode_only')
        self_s_loss = self.distance(codes, codes_aug).mean()
        return self_s_loss

    def computeCoherenceLoss(self, modalities, dataset, data, key_gt = 'image'):

        '''
        Computes the Normalized mutual information loss between the generated images for
        data batch
        :param modalities:
        :param dataset:
        :param data:
        :return:
        '''

        if len(modalities) < 2:
            ValueError("The number of sequences passed as parameter to NMI Loss must be 2")

        ot_data, ot_gen = self.loadAndRunEquivalent(data)
        dontcares = []
        for ind, i in enumerate(data['this_seq']):
            if i == data['other_seq'][ind]:
                # Since no other sequence was found, other_seq == this_seq.
                # We don't care about those.
                dontcares.append(True)
            else:
                dontcares.append(False)
        ori_gen = self.generated
        coherence_loss = self.pix2pix_model.module.criterionCoherence(ot_gen, ori_gen, dontcares = dontcares)
        coherence_loss_base =  self.pix2pix_model.module.criterionCoherence(data[key_gt],
                                                                      ot_data[key_gt], dontcares = dontcares).cuda()
        coherence_loss = self.pix2pix_model.module.coher_fact * (coherence_loss - coherence_loss_base)
        coherence_loss_base = coherence_loss_base.detach().cpu() # We detach it so that it doesn't take space

        return coherence_loss

    def loadAndRunEquivalent(self, data):

        '''
        Loads the equivalent data batch to the batch 'data' using the input dictionary 'other_seq' and 'other_image'
        by modalities. Forwards passes it through the network.
        :param data: batch of Pix2Pix_Dataset (size B)
        :return: mirrored data and result of forward pass of that data, well as data about which we don't care
        '''

        data_mirrored = data.copy()
        data_mirrored['image'] = data['im_other']
        data_mirrored['this_seq'] = data['other_seq']
        generated_mirrored = self.pix2pix_model(data_mirrored, mode='generator_no_losses')

        return data_mirrored, generated_mirrored

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_losses_nw(self):
        '''
        Get latest losses (not multiplied by any weight factors)
        :return:
        '''

        return {**self.g_losses_noweight, **self.d_losses}

    def get_disc_accuracy(self):
        return self.d_accuracy

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
        if self.mod_disc is not None:
            util.save_network(self.mod_disc, 'MD', epoch, self.opt)
        self.saveOptimizers(epoch)

    def run_tester_one_step(self, data, get_code = False, epoch = None):
        """
        Run one test step on both generators
        :param data:
        :param seqmode: The code for the sequence mode, to choose which generator to test.
        :return:
        """

        self.pix2pix_model.eval()
        self.pix2pix_model_on_one_gpu.eval()

        validation_losses = {}

        if get_code:
            g_losses, generated, accs, g_losses_nw = self.pix2pix_model(data, 'validation')
        else:
            g_losses, generated, accs, g_losses_nw, z = self.pix2pix_model(data, 'validation', get_code = True,
                                                                           epoch = epoch)

        if self.mod_disc is not None:
            loss_mod, loss_dat = self.computeModalityLoss(data, generated, validation=True)
            acc_mod, acc_data = self.computeModalityAccuracy(data, generated)
            g_losses['mod_disc'] = self.alpha_mod_disc['modality']*loss_mod.item()
            g_losses['dat_disc'] = self.alpha_mod_disc['dataset']*loss_dat.item()
            g_losses_nw['mod_disc'] = loss_mod.item()
            g_losses_nw['dat_disc'] = loss_dat.item()
            g_losses['acc_mod'] = g_losses_nw['acc_mod'] =  acc_mod.mean().item()
            g_losses['acc_dat'] = g_losses_nw['acc_dat'] = acc_data.mean().item()

        # Self-supervised encoder loss
        if self.self_supervised:
            ss_loss = self.self_supervised_lr(data, z).item()
            g_losses['self_supervised'] = ss_loss * self.ss_w
            g_losses_nw['self_supervised'] = ss_loss

        self.pix2pix_model.train()
        self.pix2pix_model_on_one_gpu.train()

        if get_code:
            return generated, g_losses, g_losses_nw, z
        else:
            return generated, g_losses, g_losses_nw


    def run_encoder_tester(self, data):
        '''
        Runs the encoder only and returns the codes
        :param data:
        :return:
        '''
        if self.opt.type_prior == 'N':
            gen_z, gen_mu, gen_logvar, noise = self.pix2pix_model(data, 'encode_only_all')
            return gen_z, gen_mu, gen_logvar, noise
        elif self.opt.type_prior == 'S':
            gen_z, gen_mu, gen_logvar  = self.pix2pix_model(data, 'encode_only_all')
            return gen_z, gen_mu, gen_logvar

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / self.opt.TTUR_factor
                new_lr_D = new_lr * self.opt.TTUR_factor

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def calculate_adaptive_weight(self, recons_loss, gener_loss, last_layer=None):
        if last_layer is not None:
            rec_grass = torch.autograd.grad(recons_loss, last_layer, retain_graph=True)[0]
            gen_grads = torch.autograd.grad(gener_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grass = torch.autograd.grad(recons_loss, self.last_layer[0], retain_graph=True)[0]
            gen_grads = torch.autograd.grad(gener_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grass) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight
        return d_weight

    def saveOptimizers(self, epoch):
        util.save_optimizer(self.optimizer_G, 'G', epoch, self.opt)
        util.save_optimizer(self.optimizer_D, 'D', epoch, self.opt)

    def loadOptimizers(self):
        util.load_optimizer(self.optimizer_G, 'G', self.opt.which_epoch, self.opt)
        util.save_optimizer(self.optimizer_D, 'D', self.opt.which_epoch, self.opt)

    def freezeModiscLayers(self):
        '''
        Freezes the relevant layers of the modality dataset discriminator,
        does the same with the normalization layers, depending on
        mod_disc_trainable.
        :return:
        '''
        for layer in self.mod_disc.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.track_running_stats = False
                layer.eval()
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.track_running_stats = False
                layer.eval()

        if not self.mod_disc_trainable:
            number_of_parameters = len(list(self.mod_disc.parameters()))
        else:
            number_of_parameters = len(list(self.mod_disc.parameters())) - 4

        for p_ind, p in enumerate(self.mod_disc.parameters()):
            if p_ind < number_of_parameters:
                p.requires_grad = False

        if self.mod_disc_trainable:
            mod_disc_params = list(self.mod_disc.parameters())[-4:]
        else:
            mod_disc_params = None

        return mod_disc_params

    def get_activation(self, name):

         def hook(model, input, output):
             self.last_hooks[name] = output.detach()

         return hook

        # How to use:
        # model.submodel.subsubmodel.register_forward_hook(get_activation(name you want))
        # print(activation(name you want))

    def registerHooks(self):


        for D in range(self.opt.num_D):
            if D == 0:
                self.pix2pix_model_on_one_gpu.netD.discriminator_0.model3.register_forward_hook(self.get_activation('disc_0'))
            elif D==1:
                self.pix2pix_model_on_one_gpu.netD.discriminator_1.model3.register_forward_hook(self.get_activation('disc_1'))
            elif D==2:
                self.pix2pix_model_on_one_gpu.netD.discriminator_2.model3.register_forward_hook(self.get_activation('disc_2'))
            elif D==3:
                self.pix2pix_model_on_one_gpu.netD.discriminator_3.model3.register_forward_hook(self.get_activation('disc_3'))
            elif D==4:
                self.pix2pix_model_on_one_gpu.netD.discriminator_4.model3.register_forward_hook(self.get_activation('disc_4'))
            elif D>5:
                ValueError("More than 5 discriminators IS NOT supported by register hooks!")

        self.pix2pix_model_on_one_gpu.netE.fc_mu.register_forward_hook(self.get_activation('enc_mu'))
        self.pix2pix_model_on_one_gpu.netE.fc_mu.register_forward_hook(self.get_activation('enc_sigma'))
        self.pix2pix_model_on_one_gpu.netG.conv_img.register_forward_hook(self.get_activation('decoder'))


