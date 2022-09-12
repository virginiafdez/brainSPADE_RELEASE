import torch

def mod_disc_pass_Accuracy(mod_disc, generated, data, modalities):
    '''
    Forward passes the output of the generator through the modality discriminator network
    and outputs the accuracy.
    :param mod_disc: modality discriminator network
    :param generated: generated images
    :param data: ground truth dictionary input
    :param modalities: list containing the modalities bound to this modality classifier.
    :return:
    '''

    # Ground truth
    targets_mod = {}  # Targets for modality classificatio
    for mod_ind, mod in enumerate(modalities):
        targets_mod[mod] = torch.zeros(len(modalities))
        targets_mod[mod][mod_ind] = 1.0

    target_mod = torch.zeros(len(data['this_seq']), len(modalities))
    for b in range(len(data['this_seq'])):
        target_mod[b, :] = targets_mod[data['this_seq'][b]]

    logits, _ = mod_disc(generated)  # We want only the modality logit.
    accuracy = 100 * (torch.argmax(torch.softmax(logits.detach().cpu(), 1), 1) ==
                      torch.argmax(target_mod, 1)).float().numpy()
    return accuracy