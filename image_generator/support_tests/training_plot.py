from moreutils import plotTrainingLosses_1dec
import os
import numpy as np
import matplotlib.pyplot as plt

folder = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC/MAY_V1_CBL_3"
file = "loss_log.txt"
save_dir = "/home/vf19/Documents/brainSPADE_2D/brainSPADE_1DEC/Training logs/training_losses_plots"
save_name = "MAY_V1_CBL_3.png"
with open(os.path.join(folder, file), 'r') as f:
    lines = f.readlines()

read = {}
relevant_lines = []
epoch_1_passed = False
header_found = False
possible_losses = []

for line in lines:
    if "Training Loss" in line:
        header_found = True
    else:
        epoch_iter = line.split(")")[0].split(",")
        epoch = int(epoch_iter[0].replace("(epoch: ", ""))
        iters = int(epoch_iter[1].replace(" iters:", ""))
        if epoch == 1 and header_found:
            read = {} # We re-start because the training was re-started and the previous entries overwritten
        if epoch not in read.keys():
            read[epoch] = {}
        header_found = False
        read[epoch][iters] = {}
        losses = line.split(")")[-1].split(";")
        for loss in losses[:-1]:
            if ")" in loss:
                loss_ = loss.split(")")[-1]
            else:
                loss_ = loss
            loss_name = loss_.strip(" ").split(":")[0]
            loss_value = float(loss_.strip(" ").strip("\n").split(":")[1])
            read[epoch][iters][loss_name] = loss_value
            if loss_name not in possible_losses:
                possible_losses.append(loss_name)

all_losses = {}
for epoch, iter in read.items():
    losses_epoch = {}
    for iter, losses in iter.items():
        for loss, loss_value in losses.items():
            if loss not in losses_epoch.keys():
                losses_epoch[loss] = [loss_value]
            else:
                losses_epoch[loss].append(loss_value)
    # Loss epoch: list of all loss values for each iteration.
    for ploss in possible_losses:
        if ploss in losses_epoch.keys():
            if ploss not in all_losses.keys():
                all_losses[ploss] = [np.mean(losses_epoch[ploss]).clip(-10, 50)]
            else:
                all_losses[ploss].append(np.mean(losses_epoch[ploss]).clip(-10, 50))
        else:
            if ploss not in all_losses.keys():
                all_losses[ploss] = [0.0]#[np.nan]
            else:
                all_losses[ploss].append(0.0)

# Prompt to know which losses you want to plot
# Each option will be 1 subplot.
options = []
options_dict = {}
for available_loss in possible_losses:
    if "D_acc" in available_loss:
        if "discrim accuracy" not in options:
            options.append("discrim accuracy")
            options_dict['discrim accuracy']="D_acc"
    elif "D_" in available_loss:
        if "discrim loss" not in options:
            options.append("discrim loss")
            options_dict['discrim loss']="D_"
    else:
        options.append(available_loss)
        options_dict[available_loss]=available_loss

input_losses = input("Input which losses you want separated by commas from the "
                     "following list:\n %s:\n" %"\n".join(options))
input_losses = input_losses.split(",")
input_losses = [i.strip(" ") for i in input_losses]

if len(input_losses) > 0:
    check = False
    while not check:
        ct=0
        print("Checking input")
        for loss in input_losses:
            if loss not in options:
                break
            else:
                ct +=1
        if ct == len(input_losses):
            check = True
        else:
            print("Wrong input.Please try again.")
            input_losses = input("Input which losses you want separated by commas from the "
                                 "following list:\n %s:\n" % "\n".join(options))
            input_losses = input_losses.split(",")
            input_losses = [i.strip(" ") for i in input_losses]


# Map back to options code names
plot_losses = []
for loss in input_losses:
    plot_losses.append(options_dict[loss])

##  #  #  Plot  #   #    ##
if len(input_losses) == 0:
    n_subplots = int(np.ceil(np.sqrt(len(possible_losses)))) # We plot every loss
else:
    n_subplots = int(np.ceil(np.sqrt(len(input_losses))))
fig = plt.figure(figsize = (12*n_subplots, 5*n_subplots))
plot_counter = 1
for loss in plot_losses:
    if loss == "D_":
        # Plot discriminator losses ont he same plot
        plt.subplot(n_subplots, n_subplots, plot_counter)
        epochs = np.arange(1, len(all_losses['D_Fake'])+1)
        plt.plot(epochs, all_losses['D_Fake'], color = "firebrick",  markersize = 5)
        plt.plot(epochs, all_losses['D_real'], color="teal",  markersize = 5)
        plt.title("Discriminator loss", fontsize = 12)
        plt.legend(["On fakes", "On reals"])
        plt.ylabel("Loss value", fontsize=11)
    elif loss == "D_acc":
        # Plot discriminator accuracies ont he same plot
        plt.subplot(n_subplots, n_subplots, plot_counter)
        epochs = np.arange(1, len(all_losses['D_Fake'])+1)
        plt.plot(epochs, all_losses['D_acc_fakes'], ":", color = "firebrick",  markersize = 0.5)
        plt.plot(epochs, all_losses['D_acc_reals'], ":", color="teal",  markersize = 0.5)
        plt.plot(epochs, all_losses['D_acc_total'], color="grey")
        plt.title("Discriminator accuracies", fontsize = 12)
        plt.legend(["On fakes", "On reals", "Total"])
        plt.ylabel("Accuracy", fontsize = 11)
    else:
        plt.subplot(n_subplots, n_subplots, plot_counter)
        epochs = np.arange(1, len(all_losses[loss])+1)
        plt.plot(epochs, all_losses[loss], color="navy", markersize = 0.5)
        plt.title("Loss %s" %loss, fontsize = 12)
        plt.ylabel("Loss value", fontsize=11)
    plt.xlabel("Epoch", fontsize = 11)
    plot_counter += 1
    plt.xticks(fontsize = 10)

plt.savefig(os.path.join(save_dir, save_name))
plt.close(fig)


