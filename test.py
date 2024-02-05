import argparse
import math
import json
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils import setup_for_distributed, MetricLogger, SmoothedValue, load_model, save_model
import models_adapter
from video_dataset import VideoDataset
import pandas as pd
from configs import DATASETS


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, help='expand_st_adapter, st_adapter')
  parser.add_argument('--classifier', type=str, help='mean, span2')
  parser.add_argument('--eval_only', action='store_true',
      help='only run evaluation.')

  parser.add_argument('--save_dir', type=str,
      help='directory to save the checkpoints in. if empty no checkpoints are saved.')
  parser.add_argument('--auto_resume', action='store_true',
      help='automatically resume from the last checkpoint.')
  parser.add_argument('--auto_remove', action='store_true',
      help='automatically remove old checkpoint after generating a new checkpoint.')
  parser.add_argument('--save_freq', type=int, default=1,
      help='save checkpoint every n epochs.')
  parser.add_argument('--resume', type=str,
      help='manually specify checkpoint to resume from. overrides --auto_resume and --pretrain.')
  parser.add_argument('--pretrain', type=str,
      help='initialize model from the given checkpoint, discard mismatching weights and '
           'do not load optimizer states.')

  parser.add_argument('--dataset', type=str, required=True)
  parser.add_argument('--mirror', action='store_true',
      help='whether mirror augmentation (i.e., random horizontal flip) should be used during training.')
  parser.add_argument('--spatial_size', type=int, default=224,
      help='spatial crop size.')
  parser.add_argument('--num_frames', type=int, default=8,
      help='number of sampled frames per video.')
  parser.add_argument('--sampling_rate', type=int, default=0,
      help='interval between sampled frames. 0 means frames evenly covers the whole video '
           '(i.e., with variable frame interval depending on the video length).)')
  parser.add_argument('--num_spatial_views', type=int, default=1, choices=[1, 3],
      help='number of spatial crops used for testing (only 1 and 3 supported currently).')
  parser.add_argument('--num_temporal_views', type=int, default=1,
      help='number of temporal crops used for testing.')
  parser.add_argument('--auto_augment', type=str,
      help='enable RandAugment of a certain configuration. see the examples in the SSv2 training scripts.')
  parser.add_argument('--num_workers', type=int, default=16,
      help='number of dataloader workers.')
  parser.add_argument('--resize_type', type=str, default='random_resized_crop',
      choices=['random_resized_crop', 'random_short_side_scale_jitter'],
      help='spatial resize type. supported modes are "random_resized_crop" and "random_short_side_scale_jitter".'
           'see implementation in video_dataset/transform.py for the details.')
  parser.add_argument('--scale_range', type=float, nargs=2, default=[0.08, 1.0],
      help='range of spatial random resize. for random_resized_crop, the range limits the portion of the cropped area; '
           'for random_short_side_scale_jitter, the range limits the target short side (as the multiple of --spatial_size).')
  parser.add_argument('--print_freq', type=int, default=10, metavar='N',
      help='print a log message every N training steps.')
  parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
      help='evaluate on the validation set every N epochs.')

  args = parser.parse_args()

  # dist.init_process_group('nccl')
  # gpu_id = dist.get_rank() % torch.cuda.device_count()
  # torch.cuda.set_device(gpu_id)
  # setup_for_distributed(dist.get_rank() == 0)
  if args.classifier:
    args.save_dir = args.save_dir + args.model_name + '_' + args.classifier + '/' + args.dataset
  else:
    args.save_dir = args.save_dir + args.model_name + '/' + args.dataset
  print('creating model')
  print(args.model_name)
  model = models_adapter.clip_vit_base_patch16_adapter12x384(args.model_name, args.classifier, num_classes=DATASETS[args.dataset]['NUM_CLASSES']).cuda().train()
  n_trainable_params = 0
  print('Total trainable params:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))
  # model = torch.nn.parallel.DistributedDataParallel(model)
  # model_without_ddp = model.module


  if args.dataset == 'hmdb_sample':
    dataframe = pd.read_csv('/hahmwj/csv_files/hmdb_sample.csv')
  elif args.dataset == 'hmdb':
    dataframe = pd.read_csv('/hahmwj/csv_files/hmdb.csv')
  elif args.dataset == 'ucf_sample':
    dataframe = pd.read_csv('/hahmwj/csv_files/ucf_sample.csv')
  elif args.dataset == 'ucf':
    dataframe = pd.read_csv('/hahmwj/csv_files/ucf.csv')
  elif args.dataset == 'k400_sample':
    dataframe = pd.read_csv('/hahmwj/csv_files/k400_sample.csv')
  elif args.dataset == 'k400':
    dataframe = pd.read_csv('/hahmwj/csv_files/k400.csv')

  print('creating dataset')
  dataset_test = VideoDataset(
      dataframe,
        'test',
      random_sample=False,
      spatial_size=args.spatial_size,
      num_frames=args.num_frames,
      sampling_rate=args.sampling_rate,
      num_spatial_views=args.num_spatial_views,
      num_temporal_views=args.num_temporal_views,
      )
  print('Test dataset:', dataset_test)
  dataloader_test = torch.utils.data.DataLoader(
      # torch.utils.data.Subset(dataset_val, range(dist.get_rank(), len(dataset_val), dist.get_world_size())),
      dataset_test,
      batch_size=1,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
      )
  weight_path = f'/hahmwj/expand_tube_st_adapter/trained_weight/{args.model_name}/{args.dataset}/checkpoint-99.pth'
  checkpoint = torch.load(weight_path, map_location='cpu')['model']
  model.load_state_dict(checkpoint, strict=False)
  def evaluate(log_stats=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    for data, labels in metric_logger.log_every(dataloader_test, 100, header):
      data, labels = data.cuda(), labels.cuda()
      B, V = data.size(0), data.size(1)
      data = data.flatten(0, 1)
      with torch.cuda.amp.autocast():
        # with model.no_sync():
        with torch.no_grad():
          logits = model(data)
        scores = logits.softmax(dim=-1)
        scores = scores.view(B, V, -1).mean(dim=1)
        acc1 = (scores.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
        acc5 = (scores.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
      metric_logger.meters['acc1'].update(acc1, n=scores.size(0))
      metric_logger.meters['acc5'].update(acc5, n=scores.size(0))
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    if log_stats is not None:
      log_stats.update({'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()})
  
  evaluate()
if __name__ == '__main__': main()
