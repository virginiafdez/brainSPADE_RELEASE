import torch
import numpy

def post_process_os(data_i):
    """
    From a SPADE Dataset item framework, extract the strings from this_seq and other_seqs to put them right after auto-collate fn call
    :param data_i: Data item obtained from pix2pix dataset
    :return:
    """

    if 'this_seq' in data_i.keys():
        var = data_i['this_seq']
        if not type(var) == str:
            data_i['this_seq'] = var[0]
    return data_i

def clear_data(trainer, data_i):

    """
    Detaches some data from the GPU to ensure that the training is possible
    :param data_i:
    :return:
    """

    data_i['image'] = data_i['image'].detach().cpu()
    if trainer is not None:
        if trainer.generated is not None:
            trainer.generated = trainer.generated.detach().cpu()

def findEquivalentPath(ori_path, folder, keywords = [], positions = [], first_come_first_served = False):

    '''
    Looks into the items contained in folder for a name matching ori_path.
    Matches are compared keyword by keyword, where the keyword is a root present between hyphens within the
    file name.
    Example:
    sub-3_ses-3_20.png (ori_path)
    Someotherfile_sub-3_ses-3_24.png (one of the items of folder)
    If keywords are "sub" and "ses", the function will output the above because in "sub-3" and "ses-3" match
    in both names.
    :param ori_path: Original path to compare to. If full path, directories are removed from it.
    :param folder:  Folder where you want to look for files.
    :param keywords: list of keywords to look for matches
    :param positions: numeric positions (hyphen-separated spaces: A_B_C (A is 0, B is 1 etc.) that match.
    :param first_come_firs_served: bool. If True, outputs the first found item, else None. If False, outputs a list
    with all matching files.
    :return:
    '''

    if "/" in ori_path:
        ori_path = ori_path.split("/")[-1]

    if not first_come_first_served:
        outputs = []

    all_files = os.listdir(folder)
    for i in all_files:
        i_sp = i.split("_")
        o_sp = ori_path.split("_")
        coincidences = []
        for ind_i, i_sub in enumerate(i_sp):
            for ind_o, o_sub in enumerate(o_sp):
                for kw in keywords:
                    if kw in i_sub and kw in o_sub:
                        if i_sub == o_sub:
                            coincidences.append(True)
                        else:
                            coincidences.append(False)
                        break
        if False not in coincidences:
            if first_come_first_served:
                return i
            else:
                outputs.append(i)

    if first_come_first_served:
        return None
    else:
        return outputs
