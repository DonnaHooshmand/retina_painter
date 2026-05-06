"""
Human in the loop deep learning segmentation for biological images

Copyright (C) 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
from os.path import dirname
sys.path.append(dirname(__file__)) # find modules in current directory

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--syncdir',
                    help=('location of directory where data is'
                           ' synced between the client and server'))
parser.add_argument('--maxbatchsize',
                    type=int,
                    default=12,
                    help='maximum batch size for training')
parser.add_argument('--model-type',
                    default='unet',
                    choices=['unet', 'retfound', 'retfound_rfa', 'fundusegmenter'],
                    help=(
                        "model backbone: 'unet' (default), 'retfound' (RETFound + plain decoder), "
                        "'retfound_rfa' (RETFound + RFA-U-Net attention decoder), "
                        "or 'fundusegmenter' (local FunduSegmenter adapter)"
                    ))

def start():
    from trainer import Trainer
    args = parser.parse_args()
    if args.syncdir:
        trainer = Trainer(sync_dir=args.syncdir,
                          max_batch_size=args.maxbatchsize,
                          model_type=args.model_type)
    else:
        trainer = Trainer(max_batch_size=args.maxbatchsize,
                          model_type=args.model_type)
    trainer.main_loop()
