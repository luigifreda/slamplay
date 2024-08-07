import logging
import yaml
import argparse
from pathlib import Path
from pprint import pformat

from models import get_model
from utils import tools  # noqa: E402
import tensorflow as tf

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('export_name', type=str)
    parser.add_argument('nms_iteration', type=int, default=2)
    args = parser.parse_args()

    export_name = args.export_name

    assert Path(args.ckpt_path).exists()
    with open(Path(args.ckpt_path, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    export_dir = Path(export_name)
    os.system("rm -rf " + export_name)

    with open(Path(args.ckpt_path, 'config.yaml'), 'r') as f:
        config['model'] = tools.dict_update(
            yaml.safe_load(f)['model'], config.get('model', {}))
    checkpoint_path = Path(args.ckpt_path, "model.ckpt-83096")
    logging.info(f'Exporting model with configuration:\n{pformat(config)}')

    config['model']['local']['nms_radius'] = 4
    config['model']['local']['num_keypoints'] = 1000
    config['model']['local']['nms_iteration'] = args.nms_iteration

    with get_model(config['model']['name'])(
            data_shape={'image': [1, None, None,
                                  1]},
            **config['model']) as net:

        net.load(str(checkpoint_path))

        tf.saved_model.simple_save(
                net.sess,
                str(export_dir),
                inputs=net.pred_in,
                outputs=net.pred_out)
        tf.train.write_graph(net.graph, str(export_dir), 'graph.pbtxt')

