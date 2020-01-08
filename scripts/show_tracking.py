import os
import os.path as osp
import sys
import numpy as np
from matplotlib.pyplot import imread, Rectangle

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))
ROOT_DIR = CURRENT_DIR
sys.path.append(ROOT_DIR)

from utils.videofig import videofig


def readbbox(file):
  with open(file, 'r') as f:
    lines = f.readlines()
    bboxs = [[float(val) for val in line.strip().replace(' ', ',').replace('\t', ',').split(',')] for line in lines]
  return bboxs


def create_bbox(bbox, color):
  return Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                   fill=False,  # remove background\n",
                   edgecolor=color)


def set_bbox(artist, bbox):
  artist.set_xy((bbox[0], bbox[1]))
  artist.set_width(bbox[2])
  artist.set_height(bbox[3])


def visualization(video_data_dir, track_log_dir):
  track_log_dir = osp.join(track_log_dir)
  # video_data_dir = osp.join(data_dir, videoname)
  te_bboxs = readbbox(osp.join(track_log_dir, 'track_rect.txt'))
  gt_bboxs = readbbox(osp.join(video_data_dir, 'groundtruth_rect.txt'))
  num_frames = len(gt_bboxs)

  def redraw_fn(ind, axes):
    ind += 1
    input_ = imread(osp.join(track_log_dir, 'image_cropped{}.jpg'.format(ind)))
    response = np.load(osp.join(track_log_dir, 'response{}.npy'.format(ind)))
    org_img = imread(osp.join(video_data_dir, 'img', '{:04d}.jpg'.format(ind + 1)))
    gt_bbox = gt_bboxs[ind]
    te_bbox = te_bboxs[ind]

    bbox = np.load(osp.join(track_log_dir, 'bbox{}.npy'.format(ind)))

    if not redraw_fn.initialized:
      ax1, ax2, ax3 = axes
      redraw_fn.im1 = ax1.imshow(input_)
      redraw_fn.im2 = ax2.imshow(response)
      redraw_fn.im3 = ax3.imshow(org_img)

      redraw_fn.bb1 = create_bbox(bbox, color='red')
      redraw_fn.bb2 = create_bbox(gt_bbox, color='green')
      redraw_fn.bb3 = create_bbox(te_bbox, color='red')

      ax1.add_patch(redraw_fn.bb1)
      ax3.add_patch(redraw_fn.bb2)
      ax3.add_patch(redraw_fn.bb3)

      redraw_fn.text = ax3.text(0.03, 0.97, 'F:{}'.format(ind), fontdict={'size': 10, },
                                ha='left', va='top',
                                bbox={'facecolor': 'red', 'alpha': 0.7},
                                transform=ax3.transAxes)

      redraw_fn.initialized = True
    else:
      redraw_fn.im1.set_array(input_)
      redraw_fn.im2.set_array(response)
      redraw_fn.im3.set_array(org_img)
      set_bbox(redraw_fn.bb1, bbox)
      set_bbox(redraw_fn.bb2, gt_bbox)
      set_bbox(redraw_fn.bb3, te_bbox)
      redraw_fn.text.set_text('F: {}'.format(ind))

  redraw_fn.initialized = False

  videofig(int(num_frames) - 1, redraw_fn,
           grid_specs={'nrows': 2, 'ncols': 2, 'wspace': 0, 'hspace': 0},
           layout_specs=['[0, 0]', '[0, 1]', '[1, :]'])

if __name__=='__main__':
  videoname = 'Boy'
  runname = 'SiamFC-3s-color-pretrained'
  video_data_dir = ROOT_DIR + '/assets/Boy'
  track_log_dir = ROOT_DIR + '/Logs/SiamFC/track_model_inference/SiamFC-3s-color-pretrained/{}'.format(videoname)
  sys.exit(visualization(video_data_dir, track_log_dir))