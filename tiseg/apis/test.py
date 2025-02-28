import mmcv
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.runner import get_dist_info
import numpy as np
import os
import cv2
from PIL import Image
import random


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random()*255 for i in range(3)]
    return r, g, b


def save_view_eval_file(pred_labeled,save_view_dir,data):
    pred_pred_label_name = '{}/{:s}_pred_label.png'.format(save_view_dir, data['metas']._data[0][0]['data_id'])
    pred_colored = np.zeros(data['metas']._data[0][0]['ori_hw']+(3,))
    pred_labeled_cnum = pred_labeled.max() + 1
    for k in range(1, pred_labeled_cnum):
        pred_colored[pred_labeled == k, :] = np.array(get_random_color())
    cv2.imwrite(pred_pred_label_name, pred_colored)


def single_gpu_test(model, data_loader,cfg=None,pre_eval=False, pre_eval_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        pre_eval (bool): Whether to evaluation by pre_eval mode. Default: False
        pre_eval_args (dict): The arguments of `def pre_eval` of dataset. Default: {}

    Returns:
        object: The processor containing results.
    """
    # when none of them is set true, return segmentation results as
    # a list of np.array.

    model.eval()
    results = []
    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    if cfg!=None:
        eval_dir=os.path.join(cfg.work_dir, 'eval')
        if cfg.save_pred==True:
            save_view_dir = os.path.join(eval_dir,"images")
            create_folder(save_view_dir)
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(**data)
            classification_pred = result[0]['classification_pred']       
        if cfg!=None and cfg.save_pred==True:
            save_view_eval_file(result[0]['inst_pred'],save_view_dir,data)
        
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices, **pre_eval_args)

        result_dict = result[0]
        result_dict['classification_F1'] = classification_pred
        result = [result_dict]
        
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model, data_loader,cfg=None,pre_eval=False, pre_eval_args={}):
    """Test model with multiple gpus by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        pre_eval_args (dict): The arguments of `def pre_eval` of dataset. Default: {}

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """

    # when none of them is set true, return segmentation results as
    # a list of np.array.

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler
    if cfg!=None:
        eval_dir=os.path.join(cfg.work_dir, 'eval')
        if cfg.save_pred==True:
            save_view_dir = os.path.join(eval_dir,"images")
            create_folder(save_view_dir)
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        if cfg!=None and cfg.save_pred==True:
            save_view_eval_file(result[0]['inst_pred'],save_view_dir,data)
        
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices, **pre_eval_args)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    # NOTE: GPU memory is really expensive
    gpu_collect = False
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), None)
    return results
