#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 16:04:03
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-14 03:57:02
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_logics_ir.commons.logging import equip_logger


@click.group(name='create')
def create():
    pass


@create.group(name='onnx')
def create_onnx():
    pass


@create_onnx.group(name='retrieve')
def create_onnx_retrieve():
    pass


@create_onnx_retrieve.command(name='huggingface')
@click.option('--mode',                  required=True,  type=click.Choice(['Model_Infos', 'Model_IDs', 'Metric_Infos', 'Task_Infos'], case_sensitive=True), help='Indicates the type of data that needs to be retrieved from Huggingface.')
@click.option('--save-dirpath',          required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--token',                 required=False, type=str, default=None, help='The HuggingFace token, which requires registering an account on HuggingFace and manually setting the access token. If None, retrieve without HuggingFace access token.')
@click.option('--mirror-url',            required=False, type=str, default='', help='The URL of the HuggingFace mirror site, which may sometimes speed up your data retrieval process, but this tools cannot guarantee data integrity of the mirror site. If not specified, use HuggingFace official site.')
@click.option('--number-per-file',       required=False, type=int, default=None, help='Used to specify the number of data items saved in each file. If None, all data will be saved in a single file.')
@click.option('--rate-limit',            required=False, type=int, default=1000, help='API rate limit: max requests in the last 5 minutes. Default: 1000. Interval is calculated as 300s / (rate_limit * 0.9).')
@click.option('--include-storage',       is_flag=True,   help='For Model_Infos mode: include usedStorage field for each model (requires additional API calls). Default: skip storage retrieval to save API quota.')
@click.option('--logging-filepath',      required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_retrieve_huggingface(
    mode,
    save_dirpath,
    token,
    mirror_url,
    number_per_file,
    rate_limit,
    include_storage,
    logging_filepath,
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.hubs.huggingface import retrieve

    kwargs = dict(
        token=token,
        number_per_file=number_per_file,
        rate_limit=rate_limit,
        include_storage=include_storage,
    )

    retrieve.main(mode, save_dirpath, mirror_url, **kwargs)


@create_onnx_retrieve.command(name='onnx')
@click.option('--mode',             required=True,  type=click.Choice(['Model_Infos', 'Model_IDs'], case_sensitive=True), help='Indicates the type of data that needs to be retrieved from ONNX Model Zoo.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--force-reload',     is_flag=True,   help='Use to ignore previous download records and redownload after use.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_retrieve_onnx(
    mode,
    save_dirpath,
    force_reload,
    logging_filepath,
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.hubs.onnx import retrieve

    kwargs = dict(
        force_reload=force_reload,
    )

    retrieve.main(mode, save_dirpath, **kwargs)


@create_onnx_retrieve.command(name='torch')
@click.option('--mode',             required=True,  type=click.Choice(['Model_Infos', 'Model_IDs'], case_sensitive=True), help='Indicates the type of data that needs to be retrieved from Torch Hub.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--force-reload',     is_flag=True,   help='Use to ignore previous download records and redownload after use.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_retrieve_torch(
    mode,
    save_dirpath,
    force_reload,
    logging_filepath,
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.hubs.torch import retrieve

    kwargs = dict(
        force_reload=force_reload,
    )

    retrieve.main(mode, save_dirpath, **kwargs)


@create_onnx.group(name='convert')
def create_onnx_convert():
    pass


@create_onnx_convert.command(name='huggingface')
@click.option('--model-infos-filepath', required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model Infos file, which is obtained using the command: `younger logics ir create onnx retrieve huggingface --mode Model_Infos ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--device',               required=False, type=click.Choice(['cpu', 'cuda'], case_sensitive=True), default='cpu', help='Used to indicate whether to use GPU or CPU when converting models.')
@click.option('--framework',            required=False, type=click.Choice(['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], case_sensitive=True), default='optimum', help='Indicates the framework to which the model belonged prior to conversion.')
@click.option('--token',                required=False, type=str, default=None, help='The HuggingFace token, which requires registering an account on HuggingFace and manually setting the access token. If None, retrieve without HuggingFace access token.')
@click.option('--estimate',             is_flag=True,   help='Use to estimate models will be converted. No Conversion Processes.')
@click.option('--logging-filepath',     required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_convert_huggingface(
    model_infos_filepath,
    save_dirpath, cache_dirpath,
    device, framework, token, estimate,
    logging_filepath
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.hubs.huggingface import convert

    convert.main(model_infos_filepath, save_dirpath, cache_dirpath, device=device, framework=framework, token=token, estimate=estimate)


@create_onnx_convert.command(name='onnx')
@click.option('--model-infos-filepath', required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model Infos file, which is obtained using the command: `younger logics ir create onnx retrieve onnx --mode Model_Infos ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--logging-filepath',     required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_convert_onnx(
    model_infos_filepath,
    save_dirpath, cache_dirpath,
    logging_filepath
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.hubs.onnx import convert

    convert.main(model_infos_filepath, save_dirpath, cache_dirpath)


@create_onnx_convert.command(name='torch')
@click.option('--model-infos-filepath', required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model Infos file, which is obtained using the command: `younger logics ir create onnx retrieve torch --mode Model_Infos ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--logging-filepath',     required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def create_onnx_convert_torch(
    model_infos_filepath,
    save_dirpath, cache_dirpath,
    logging_filepath
):
    equip_logger(logging_filepath=logging_filepath)

    from younger_logics_ir.scripts.hubs.torch import convert

    convert.main(model_infos_filepath, save_dirpath, cache_dirpath)


@create.group(name='code')
def create_code():
    pass


@create_code.group(name='convert')
def create_code_convert():
    pass