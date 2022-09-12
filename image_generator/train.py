'''
Trainer of a multi-sequence SPADE network with NMI Loss to ensure cross-sequence coherence
Author: Virginia Fernandez
'''
import os
import moreutils as uvir
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from data.dataset_utils import clear_data
import numpy as np
import torch
from trainers.pix2pix_trainer import Pix2PixTrainer
from copy import deepcopy
import gc
from data.spadenai_v2 import SpadeNai
from data.spadenai_v2_sliced import SpadeNaiSlice
import shutil
from monai.data.dataloader import DataLoader
from util.tensorboard_writer import BrainspadeBoard


plot_errors = False
# Parse options
opt = TrainOptions().parse()

# # Save images for discriminator training
# folder_save = "/home/vf19/Documents/brainSPADE_2D/DATA/DISCRIMINATOR_ONLY_TRAINING_VAL"
# if not os.path.isdir(folder_save):
#     os.makedirs(folder_save)

# Remove triplet
os.chdir('..')  # Change directory to the previous one

# Dataset
if opt.dataset_type == 'sliced':
    dataset_container = SpadeNaiSlice(opt, mode = 'train')
    dataset_val_container = SpadeNaiSlice(opt, mode='validation')
else:
    dataset_container = SpadeNai(opt, mode = 'train')
    dataset_val_container = SpadeNai(opt, mode = 'validation', store_and_use_slices=True)
dataloader = DataLoader(dataset_container.sliceDataset,
                                         batch_size=opt.batchSize, shuffle=False,
                                         num_workers=int(opt.nThreads), drop_last=opt.isTrain)
dataloader_val = DataLoader(dataset_val_container.sliceDataset,
                                             batch_size=opt.batchSize, shuffle=False,
                                             num_workers=int(opt.nThreads), drop_last=False)

# Initialisation network
trainer = Pix2PixTrainer(opt)

# Iterations counter
iter_counter = IterationCounter(opt, len(dataset_container))

# Visualization tool
visualizer = Visualizer(opt)
visualizer.initialize_Validation(opt.continue_train)
if visualizer.back_up_validation_slices() and opt.dataset_type == 'volume':
    dataset_val_container.read_stored_slices(store=os.path.join(visualizer.val_dir))

# Tensorboard
if opt.use_tboard:
    tboard = BrainspadeBoard(opt)

# Validation save ID
save_im_id = None

# Gradients saved
gradients = {}
activations = {}

# Debug dataloader ***
# for epoch in range(100):
#     print("Epoch %d/100" %(epoch))
#     try:
#         for dind, data_i in enumerate(dataloader):
#             pass
#     except Exception as e:
#         print("Exception %d: %s" %(dind, str(e)))

# Training Loop
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    train_gen_count = 0
    train_dis_count = 0
    train_total = 0
    for dind, data_i in enumerate(dataloader):
        train_total += 1

        # Phase 1 Generator training.
        # If accuracy is none, we train.
        # If accuracy is < 65%, we train discriminator only.
        # If accuracy is > 85%, we train generator only.
        # If accuracy is between both, we train both.

        # Ammend dataset container to register stored slices
        if dataset_container.diff_style_volume and dataset_container.store_slices:
            dataset_container.compute_slices(data_i['label_path'], data_i['slice_no'],
                                             data_i['style_label_file'], data_i['slice_style_no'])
        elif dataset_container.store_slices:
            dataset_container.compute_slices(data_i['label_path'], data_i['slice_no'],
                                             data_i['label_path'], data_i['slice_style_no'])
        if dataset_container.intensify_lesions['flag'] and opt.dataset_type == 'volume':
            dataset_container.compute_lesions()

        d_acc = trainer.get_disc_accuracy()
        if d_acc is None:
            d_acc = iter_counter.add_assess_accuracy(None)
        else:
            d_acc = iter_counter.add_assess_accuracy(d_acc['D_acc_total'])
        if d_acc is None:
            train_disc = True
            train_gen = True
        elif trainer.disc_threshold['low'] <= d_acc <= trainer.disc_threshold['up']:
            train_disc = True
            train_gen = True
        elif d_acc < trainer.disc_threshold['low']:
            train_disc = True
            train_gen = False
        elif d_acc > trainer.disc_threshold['up']:
            train_disc = False
            train_gen = True
        else:
            Warning("Non numeric accuracy.")

        iter_counter.record_one_iteration()

        # Phase 1 Generator training.
        if train_gen:
            # If train_enc_only < epochs it means that from this epoch onward we only
            # train the encoder. Otherwise, both.
            iter_counter.record_one_gradient_iteration_gen()
            if opt.train_enc_only is not None:
                if epoch >= opt.train_enc_only:
                    trainer.run_encoder_one_step(data_i)
                else:
                    if iter_counter.needs_gradient_calc(for_disc=False) and opt.tboard_gradients:
                        if opt.topK_discrim:
                            gradients_ = trainer.run_generator_one_step(data_i, with_gradients=True, epoch = epoch)
                        else:
                            gradients_ = trainer.run_generator_one_step(data_i, with_gradients=True)
                        gradients.update(gradients_)
                    else:
                        if opt.topK_discrim:
                            trainer.run_generator_one_step(data_i, epoch = epoch)
                        else:
                            trainer.run_generator_one_step(data_i)
            else:
                if iter_counter.needs_gradient_calc(for_disc=False) and opt.tboard_gradients:
                    if opt.topK_discrim:
                        gradients_ = trainer.run_generator_one_step(data_i, with_gradients=True, epoch = epoch)
                    else:
                        gradients_ = trainer.run_generator_one_step(data_i, with_gradients=True)
                    gradients.update(gradients_)
                else:
                    if opt.topK_discrim:
                        trainer.run_generator_one_step(data_i, epoch=epoch)
                    else:
                        trainer.run_generator_one_step(data_i)
            if iter_counter.needs_activations(for_disc=False) and opt.tboard_activations:
                activations.update({'enc_mu': trainer.last_hooks['enc_mu'],
                                    'enc_sigma': trainer.last_hooks['enc_sigma'],
                                    'deocder': trainer.last_hooks['decoder']})

            generated = trainer.get_latest_generated()
            #uvir.saveBatchImages(data_i, generated, folder_save)

            # If display is needed, we save the relevant data.
            data_copy = deepcopy(data_i)

            # Store generator losses
            iter_counter.store_losses(trainer.g_losses, None)

            # Part 1-D. If modality discriminator is active, save the images with highest errors.
            if plot_errors:
                if trainer.checkDiscriminatorLosses():
                    loss_modisc = trainer.g_losses['mod_disc'].item()
                    loss_modisc = np.round(loss_modisc, 5)
                    # Obtain latest generated images
                    generated = trainer.get_latest_generated()
                    img_dir = visualizer.check_errors_dir
                    fig_name = os.path.join(img_dir,
                                            "epoch_%s_iter_%s.png" % (epoch, iter_counter.total_steps_so_far))
                    all_to_save = []
                    all_to_save.append(data_i['label']) # B x C
                    all_to_save.append(data_i['style_image'])
                    all_to_save.append(generated)
                    titles = ["Input Label", "Input Style", "Generated (error)"]
                    b_acc = {'sequence': data_copy['this_seq'],
                             'error': [str(loss_modisc)]*len(data_copy['this_seq'])}
                    uvir.saveFigs(all_to_save, fig_name, create_dir= True, nlabels= opt.label_nc,
                                  same_scale = True, titles = titles, batch_accollades = b_acc,
                                  index_label = 0, bound_normalization=opt.bound_normalization)

            train_gen_count += 1

        # Part 2. Train the discriminator.
        if train_disc:

            iter_counter.record_one_gradient_iteration_dis()
            if iter_counter.needs_gradient_calc(for_disc=True) and opt.tboard_gradients:
                if opt.topK_discrim:
                    if iter_counter.needs_D_display():
                        gradients_, outputs_D = trainer.run_discriminator_one_step(data_i,
                                                                                   with_gradients=True, epoch = epoch,
                                                                                   return_predictions=True)
                    else:
                        gradients_ = trainer.run_discriminator_one_step(data_i, with_gradients=True, epoch = epoch)
                else:
                    if iter_counter.needs_D_display():
                        gradients_, outputs_D = trainer.run_discriminator_one_step(data_i, with_gradients=True,
                                                                                   return_predictions=True)
                    else:
                        gradients_ = trainer.run_discriminator_one_step(data_i, with_gradients=True)
                gradients.update(gradients_)
            else:
                if opt.topK_discrim:
                    if iter_counter.needs_D_display():
                        outputs_D = trainer.run_discriminator_one_step(data_i, epoch = epoch, return_predictions=True)
                    else:
                        trainer.run_discriminator_one_step(data_i, epoch = epoch)
                else:
                    if iter_counter.needs_D_display():
                        outputs_D = trainer.run_discriminator_one_step(data_i, epoch = epoch, return_predictions=True)
                    else:
                        trainer.run_discriminator_one_step(data_i)
            iter_counter.store_losses(None, trainer.d_losses, trainer.d_accuracy)
            train_dis_count += 1

            if iter_counter.needs_activations(for_disc=True) and opt.tboard_activations:
                for key_hook, val_hook in trainer.last_hooks.items():
                    if 'disc' in key_hook:
                        activations.update({key_hook:val_hook})


        # Part 3. Tests and display
        # Part 3-1. Code distribution boxplots are saved in web/code_plots
        if iter_counter.needs_enc_display():
            if opt.type_prior == 'N':
                gen_z, gen_mu, gen_logvar, gen_noise = trainer.run_encoder_tester(data_i)
                visualizer.save_codes(gen_z, gen_mu, gen_logvar, data_i['this_seq'], gen_noise, iter_counter.current_epoch,
                                 iter_counter.epoch_iter)
            elif opt.type_prior == 'S':
                gen_z, gen_mu, gen_logvar = trainer.run_encoder_tester(data_i)
                visualizer.save_codes(gen_z, gen_mu, gen_logvar, sequence = data_i['this_seq'],
                                      epoch = iter_counter.current_epoch,
                                      iter = iter_counter.epoch_iter)
        if iter_counter.needs_D_display():
            visualizer.plot_D_results(outputs_D, epoch, iter_counter.epoch_iter)


        # Clear.
        clear_data(trainer, data_i)

        # Part 3-2. Printing of losses IF:
        # We are every N_print iterations
        # We have latest losses.

        if iter_counter.needs_printing() and iter_counter.epoch_iter>1:
            if trainer.g_losses is not None:
                losses = trainer.get_latest_losses_nw()
                accuracies = trainer.get_disc_accuracy()
                losses.update(accuracies)
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                     losses, iter_counter.time_per_iter)
            else:
                print("No losses available at printing time for epoch %d and iteration %d" %(epoch,
                                                                                             iter_counter.epoch_iter))

        # Part 3-3. We save the last generated set of mages in web/images
        if iter_counter.needs_displaying() and iter_counter.epoch_iter>1:
            if trainer.get_latest_generated() is not None:
                generated = trainer.get_latest_generated()
                img_dir = visualizer.img_dir
                fig_name = os.path.join(img_dir,
                                        "epoch_%s_iter_%s.png" % (epoch, iter_counter.total_steps_so_far))
                # We append all we want to save in a list
                all_to_save = []
                b_ind = np.random.randint(0, len(data_copy['this_seq'])) # We select one of the batch items
                all_to_save.append(data_copy['label'])
                all_to_save.append(torch.cat([data_copy['style_image'][:, 0, :, :].unsqueeze(1)] * 3, dim=1))
                all_to_save.append(torch.cat([data_copy['image'][:, 0, :, :].unsqueeze(1)]*3, dim = 1))
                all_to_save.append(generated)
                titles = ["Input Label", "Input style", "Ground truth",  "Generated (sequence)"]
                b_acc = {'sequence': data_copy['this_seq'][b_ind]}
                uvir.saveFigs(all_to_save, fig_name, create_dir=True, nlabels = opt.label_nc, index=b_ind,
                              same_scale= True, titles = titles, batch_accollades = b_acc, index_label = 0,
                              bound_normalization=opt.bound_normalization)

        # Part 3-4. We save the iteration stage of the network.
        if iter_counter.needs_saving():
            # Save data
            print('Saving the latest model (epoch %d, total steps %d)' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()


        # Part 3-5. If gradients, then plot summary them
        if len(gradients) != 0:
            try:
                if opt.use_tboard and opt.tboard_gradients:
                    tboard.log_grad_histograms(gradients, epoch, iter_counter.epoch_iter,
                                               len(dataset_container)*opt.nThreads)
            except:
                gradients = {}

        # Part 3.6. If activations, then plot summary them
        if len(activations) != 0:
            if opt.use_tboard and opt.tboard_activations:
                tboard.log_act_histograms(activations, epoch, iter_counter.epoch_iter,
                                           len(dataset_container)*opt.nThreads)
            activations = {}


    torch.cuda.empty_cache()  # Epoch end. Empty cache


    # Part 4. Validation.
    if iter_counter.needs_testing():
        print("Validation results:\n")
        if opt.z_dim in [2,3]:
            do_code = True # Plot codes of each validation image in 2D or 3D plot.
        else:
            do_code = False # If the latent dimension is > 3, we don't plot (dimensionality reduction would be required)
        with torch.no_grad():
            # We select only one image from the validation set
            if dataset_val_container.__len__() > 1:
                if save_im_id == None:
                    save_im_id = np.random.randint(0, dataset_val_container.__len__() - 1)
            else:
                save_im_id = 0

            # Epoch-wise metrics to-validation
            results_values = []  # Values for the test
            accuracy_mod = []
            accuracy_dat = []
            losses_nw = {}
            losses = {}
            if do_code:
                codes = {}

            for t, data_t in enumerate(dataloader_val):
                if opt.dataset_type =='volume':
                    dataset_val_container.compute_slices(data_t['label_path'], data_t['slice_no'],
                                                         data_t['style_label_path'], data_t['slice_style_no'])
                if do_code:
                    gen_val, g_losses, g_losses_nw, code_val = trainer.run_tester_one_step(data_t, get_code=True)
                    # Store codes
                    for b in range(code_val.shape[0]):
                        if data_t['this_seq'][b]+"-"+data_t['st_dataset'][b] in codes.keys():
                            codes[data_t['this_seq'][b]+"-"+data_t['st_dataset'][b]].append(code_val[b,...].detach().cpu())
                        else:
                            codes[data_t['this_seq'][b]+"-"+data_t['st_dataset'][b]] = [code_val[b,...].detach().cpu()]
                    # Store accuracies
                    if 'acc_mod' in g_losses_nw.keys() and 'acc_dat' in g_losses_nw.keys():
                        accuracy_mod.append(g_losses_nw['acc_mod'])
                        accuracy_dat.append(g_losses_nw['acc_dat'])
                else:
                    gen_val, g_losses, g_losses_nw = trainer.run_tester_one_step(data_t, epoch=epoch)
                    # Store accuracies
                    if 'acc_mod' in g_losses_nw.keys() and 'acc_dat' in g_losses_nw.keys():
                        accuracy_mod.append(g_losses_nw['acc_mod'])
                        accuracy_dat.append(g_losses_nw['acc_dat'])

                clear_data(None, data_t)

                # Part 4-1 Save images in web/validation
                if t == save_im_id:  # We only save one of the instances per epoch
                    fig_name = os.path.join(visualizer.val_dir,
                                            "validation_epoch_%s_%s.png" % ('inference', epoch))
                    val_imgs = [data_t['label'], data_t['style_image'], data_t['image'], gen_val]
                    titles = ["Input Label", "Input style", "GT", "Synth. (sequence)"]
                    b_acc = {'sequence':data_t['this_seq']}

                    uvir.saveFigs(val_imgs, fig_name, create_dir=True, nlabels = opt.label_nc,
                                  same_scale= True, titles = titles, batch_accollades= b_acc,
                                  index_label= 0, bound_normalization=opt.bound_normalization)

                # Image quality metric
                ssim_item = 0
                ssim_item += uvir.structural_Similarity(data_t['image'], gen_val, mean=True)
                results_values.append(ssim_item)

                # Average losses
                for loss_item, loss_value in g_losses_nw.items():
                    if 'acc_mod' in loss_item or 'acc_dat' in loss_item:
                        # These are treated separately!
                        continue
                    # Unweighted losses
                    if loss_item not in losses_nw.keys():
                        losses_nw[loss_item] = [loss_value]
                    else:
                        losses_nw[loss_item].append(loss_value)
                    # Weighted loss
                    if loss_item not in losses.keys():
                        losses[loss_item] = [g_losses[loss_item]]
                    else:
                        losses[loss_item].append(g_losses_nw[loss_item])

            # Plot codes if requested
            if do_code:
                uvir.plotCodes(codes, opt.sequences, opt.datasets, opt.z_dim,
                               os.path.join(visualizer.val_dir, 'code_plots_%d.png' %epoch),
                               epoch)

            # Process losses and register them
            final_losses = {}
            final_losses_nw = {}
            for loss_item, loss_value_list in losses.items():
                final_losses[loss_item] = np.mean(loss_value_list)
                final_losses_nw[loss_item] = np.mean(losses_nw[loss_item])

            if opt.dataset_type == 'volume':
                if not visualizer.back_up_validation_slices():
                    dataset_val_container.back_up_stored_slices(store = os.path.join(visualizer.val_dir))
            visualizer.register_Val_Losses(epoch, errors_nw=final_losses_nw, errors_w=final_losses, print_it=True)

        # Part 4-2 Save structural similarity txt in web/validation
        if len(accuracy_mod) == 0:
            accuracy_mod = [-1]
        if len(accuracy_dat) == 0:
            accuracy_dat = [-1]
        visualizer.register_Test_Results(epoch, {'SSIM': np.mean(results_values), 'Modality-Disc': np.mean(accuracy_mod),
                                          'Dataset-Disc': np.mean(accuracy_dat)})
        print("SSIM %.3f\tAcc_mod %.3f\tAcc_data %.3f\n" %(100*np.mean(results_values),
                                                                         np.mean(accuracy_mod),
                                                                         np.mean(accuracy_dat)))

    # Part 5. We update LR
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    # Part 6. We save the network.
    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
    if epoch % opt.save_epoch_copy == 0:
        trainer.save(epoch)

    print("Trained generator %d/%d" %(train_gen_count, train_total))
    print("Trained discriminator %d/%d" %(train_dis_count, train_total))

    # Part 7. Summary writer
    if opt.use_tboard:
        tboard.log_results(iter_counter.getStoredLosses(), epoch, is_val=False)
        tboard.log_results(final_losses_nw, epoch, is_val=True)

    # Part 8. Cleaning.
    gc.collect()
    torch.cuda.empty_cache()

print("Removing cache directory...")
shutil.rmtree(dataset_container.cache_dir)
