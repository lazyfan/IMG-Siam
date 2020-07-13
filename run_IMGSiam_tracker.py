import logging
import os
import os.path as osp
import sys
from glob import glob
import tensorflow as tf

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s - %(message)s')

import configuration
from inference import inference_wrapper
from inference.tracker import Tracker
from utils.infer_utils import Rectangle
from utils.misc_utils import auto_select_gpu, mkdir_p, sort_nicely, load_cfgs
from scripts.show_tracking_result import visualization

CHECKPOINT = ROOT_DIR + '/Logs/model_checkpoints/IMG-Siam'  # IMGSiam-3s-color
VIDEO_FILE = ROOT_DIR + '/assets/video'
# VIDEO_FILE = ROOT_DIR + '/assets/video.mp4'

def main(checkpoint, video_file):
  os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

  if 'model_config.json' in os.listdir(CHECKPOINT) and 'track_config.json' in os.listdir(CHECKPOINT):
    model_config, _, track_config = load_cfgs(checkpoint)
  else:
    model_config = configuration.MODEL_CONFIG
    track_config = configuration.TRACK_CONFIG
  if track_config['visualization']:
    track_config['log_level'] = 1

  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
  g.finalize()

  if not osp.isdir(track_config['log_dir']):
    logging.info('Creating inference directory: %s', track_config['log_dir'])
    mkdir_p(track_config['log_dir'])

  logging.info("Running tracking on %d videos matching %s", len(video_file), video_file)

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(graph=g, config=sess_config) as sess:
    restore_fn(sess)

    tracker = Tracker(model, model_config=model_config, track_config=track_config)

    if not osp.isdir(video_file):
      logging.warning('{} is not a directory, skipping...'.format(video_file))

    video_name = osp.basename(video_file)
    video_log_dir = osp.join(track_config['log_dir'], video_name)
    mkdir_p(video_log_dir)

    filenames = sort_nicely(glob(video_file + '/img/*.jpg'))
    first_line = open(video_file + '/first_frame_rect.txt').readline()
    bb = [int(v) for v in first_line.strip().split(',')]
    init_bb = Rectangle(bb[0] - 1, bb[1] - 1, bb[2], bb[3])  # 0-index in python
    matting_method = track_config['matting']

    trajectory = tracker.track(sess, init_bb, filenames, matting_method, video_log_dir)
    with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
      for region in trajectory:
        rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
                                          region.width, region.height)
        f.write(rect_str)

  if track_config['visualization']:
    logging.info("Viewing visualization results in a new window.")
    visualization(video_file, video_log_dir)


if __name__ == '__main__':
  sys.exit(main(CHECKPOINT, VIDEO_FILE))