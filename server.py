import numpy as np
from PIL import Image
import dnnlib

from flask import Flask, request, send_file
from flask_cors import cross_origin
from os import environ, makedirs, path, remove
from subprocess import run
from glob import glob
from math import sqrt
from shutil import rmtree

import run_generator

from google.cloud import storage

# connect to storage to place morphs
client = storage.Client()
bucket = client.get_bucket('botany-morphs')

app = Flask(__name__)

network = environ.get('NETWORK', None)
if network == None:
    print("Set env var NETWORK to stylegan pkl")
    exit()


def kwargs_from_projections(image_1, image_2, frames):
    projection_patt = 'w_latents/{}*.npy'
    npy_1 = glob(projection_patt.format(image_1))[0]
    npy_2 = glob(projection_patt.format(image_2))[0]

    ws_1 = run_generator._parse_npy_files(npy_1)
    ws_2 = run_generator._parse_npy_files(npy_2)

    return {
        'network_pkl': network,
        'truncation_psi': 1.0,
        'walk_type': 'line-w',
        'frames': frames-1,
        'npys': [ws_1, ws_2],
        'npys_type': 'w',
        'result_dir': 'server_walk_results',
        'seeds': [],
        'save_vector': False
    }


def kwards_from_seeds(image_1: str, image_2, frames):
    seed1 = image_1.split("seed")[1].split(".png")[0].split(".jpg")[0]
    seed2 = image_2.split("seed")[1].split(".png")[0].split(".jpg")[0]

    return {
        'network_pkl': network,
        'truncation_psi': 1.0,
        'walk_type': 'line',
        'frames': frames-1,
        'result_dir': 'server_walk_results',
        'seeds': [seed1, seed2],
        'save_vector': False
    }


@app.route("/stylegan/morph", methods=["POST"])
@cross_origin()
def morph():
    image_1 = request.json['image1']
    image_2 = request.json['image2']
    fc = request.json['frame_count']
    no_cache = 'no_cache' in request.json and request.json['no_cache'] is True
    seeded = 'seeded' in request.json and request.json['seeded'] is False
    if fc is None:
        fc = 9
    # floor the resolution
    frames = int(sqrt(fc)) ** 2

    # ensure that our morphs are always alphabetically sorted
    image_1, image_2 = sorted([image_1, image_2])
    output_path = "spritesheets/{}_{}_{}.jpg".format(image_1, image_2, frames)
    # check if sheet exists to return early
    bucket_blob = bucket.blob(output_path)
    if not no_cache and bucket_blob.exists():
        return bucket_blob.public_url

    if seeded:
        kwargs = kwards_from_seeds()
    else:
        try:
            kwargs = kwargs_from_projections()
        except IndexError:
            return "projection for one or both images not found", 400

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = "serve-latent-walk"
    result_path, _ = dnnlib.submit_run(
        sc, 'run_generator.generate_latent_walk', **kwargs)
    morph_pattern = "{}/*.png".format(result_path)
    sheet = make_spritesheet(morph_pattern, output_path)
    # delete result_path since we have our sheet
    rmtree(result_path)
    # upload to bucket instead of storing locally
    bucket_blob.upload_from_filename(sheet)
    remove(sheet)
    # return the file
    return bucket_blob.public_url


def make_spritesheet(pattern, output_path):
    # make sure output dir exists
    makedirs("spritesheets", exist_ok=True)
    # count images
    result_imgs = glob(pattern)
    edge_count = sqrt(len(result_imgs))
    assert(edge_count % 1 == 0)  # ensure we have a square number
    # run image magick
    # montage *.png -geometry 512x512 -colors 32 spritesheets/example.png
    run([
        "montage", pattern, "-geometry", "512x512", "-colors", "32", output_path
    ])
    return output_path
