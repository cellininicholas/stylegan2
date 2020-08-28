import numpy as np
import PIL.Image
import dnnlib

from flask import Flask, request
from flask_cors import cross_origin
from os import environ
from glob import glob

import run_generator

app = Flask(__name__)

network = environ.get('NETWORK', None)
if network == None:
    print("Set env var NETWORK to stylegan pkl")
    exit()


@app.route("/stylegan/morph", methods=["POST"])
@cross_origin()
def morph():
    image_1 = request.json['image1']
    image_2 = request.json['image2']

    # ensure that our morphs are always alphabetically sorted
    image_1, image_2 = sorted([image_1, image_2])

    projection_patt = 'w_latents/{}*.npy'
    npy_1 = glob(projection_patt.format(image_1))[0]
    npy_2 = glob(projection_patt.format(image_2))[0]

    ws_1 = run_generator._parse_npy_files(npy_1)
    ws_2 = run_generator._parse_npy_files(npy_2)

    kwargs = {
        'network_pkl': network,
        'truncation_psi': 1.0,
        'walk_type': 'line-w',
        'frames': 25,
        'npys': [ws_1, ws_2],
        'npys_type': 'w',
        'result_dir': 'server_walk_results',
        'seeds': [],
        'save_vector': False
    }

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = "serve-latent-walk"
    dnnlib.submit_run(
        sc, 'run_generator.generate_latent_walk', **kwargs)
