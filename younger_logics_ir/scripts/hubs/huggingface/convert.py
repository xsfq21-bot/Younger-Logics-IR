#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-09-09 14:40:09
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import onnx
import tqdm
import pathlib
import multiprocessing

from typing import Any, Literal, Callable

from huggingface_hub import login, hf_hub_download, snapshot_download, utils, HfApi

from younger.commons.io import saves_json, loads_json, save_json, load_json, create_dir, delete_dir, get_human_readable_size_representation
from younger.commons.hash import hash_string

from younger_logics_ir.modules import Instance, Implementation, Origin
from younger_logics_ir.converters import convert
from younger_logics_ir.converters.onnx2ir.io import load_model, check_model

from younger_logics_ir.commons.logging import logger
from younger_logics_ir.commons.constants import YLIROriginHub

from younger_logics_ir.scripts.commons.utils import get_onnx_opset_versions, get_onnx_model_opset_version

from .utils import get_huggingface_hub_model_readme, get_huggingface_hub_model_siblings, clean_huggingface_hub_model_cache, infer_supported_frameworks, is_permanent_error, get_minimum_opset_from_error


def clean_cache(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path):
    delete_dir(cvt_cache_dirpath, only_clean=True)
    clean_huggingface_hub_model_cache(model_id, ofc_cache_dirpath)


def safe_optimum_export(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue, device: str, library_name: str | None = None):
    import os
    import inspect

    # Redirect stdout and stderr to /dev/null BEFORE importing optimum/torch,
    # otherwise torch registration warnings leak to the parent terminal.
    # All communication back to the parent process goes through the results_queue.
    
    # saved_fds = {1: os.dup(1), 2: os.dup(2)}
    # devnull = os.open(os.devnull, os.O_WRONLY)
    # os.dup2(devnull, 1)
    # os.dup2(devnull, 2)
    # os.close(devnull)

    saved_fds = {}
    fd_redirected = False

    try:
        try:
            saved_fds = {1: os.dup(1), 2: os.dup(2)}
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            fd_redirected = True
        except OSError:
            pass

        try:
            from optimum.exporters.onnx import main_export

            export_kwargs = dict(
                opset=onnx_opset_version,
                device=device,
                cache_dir=ofc_cache_dirpath,
                monolith=True,
                do_validation=False,
                trust_remote_code=True,
                no_post_process=True,
                library_name=library_name,
            )

            # Why switch by opset:
            # - Optimum docs recommend dynamo exporter for opset >= 18, while
            #   opset < 18 can keep the legacy TorchScript exporter path.
            # - TorchScript is part of PyTorch; no standalone TorchScript package
            #   is required.
            # Ref: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model
            # Keep a guard for environments where main_export has no `dynamo`
            # parameter (older optimum versions).

            if 'dynamo' in inspect.signature(main_export).parameters:
                export_kwargs['dynamo'] = (onnx_opset_version >= 18)

            # Why not enable custom export knobs here by default:
            # The custom path (model_kwargs/custom_onnx_configs/fn_get_submodels)
            # is model-family-specific and may require per-architecture config.
            # The current pipeline favors broad, stable batch conversion.
            # Ref: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model#customize-the-export-of-official-transformers-models

            main_export(model_id, cvt_cache_dirpath, **export_kwargs)
            this_status = 'success'
            this_error = ''
        except MemoryError as exception:
            this_status = 'oversize'
            this_error = str(exception)
        except utils.RepositoryNotFoundError as exception:
            this_status = 'access_deny'
            this_error = str(exception)
        except Exception as exception:
            error_text = f"{type(exception).__name__}: {exception}"

            if (
                type(exception).__name__ == "OutOfMemoryError"
                or "CUDA out of memory" in error_text
                or "out of memory" in error_text.lower()
            ):
                this_status = "oversize"
            else:
                this_status = "convert_error"

            this_error = error_text
    except Exception as exception:
        this_status = "covert_worker_error"
        this_error = f"{type(exception).__name__}: {exception}"

    finally:
        if fd_redirected:
            for fd, saved in saved_fds.items():
                try:
                    os.dup2(saved, fd)
                    os.close(saved)
                except OSError:
                    pass

        results_queue.put((this_status, this_error))


def convert_optimum(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None) -> tuple[dict[int, tuple[Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill'], dict[str, Literal['success', 'logicx_error']]]], list[Instance], list[str]]:
    assert device in {'cpu', 'cuda'}
    status: dict[int, tuple[Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill'], dict[str, Literal['success', 'logicx_error']]]] = dict()
    instances: list[Instance] = list()
    artifacts: list[str] = list()

    # The highest opset version supported by torch.onnx.export is 20
    # torch.onnx.export & optimum.onnx.main_export decide the highest supported opset version.
    # TODO: It is a complex decision to determine the highest supported opset version accurately.
    # TODO: Please make this decision more robust and accurate.
    from torch.onnx import _constants
    opset_versions = [v for v in get_onnx_opset_versions() if v <= _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET]
    opset_index = 0
    while opset_index < len(opset_versions):
        onnx_opset_version = opset_versions[opset_index]
        results_queue = multiprocessing.Queue()
        subprocess = multiprocessing.Process(target=safe_optimum_export, args=(model_id, cvt_cache_dirpath, ofc_cache_dirpath, onnx_opset_version, results_queue, device, library_name))
        subprocess.start()
        subprocess.join()

        this_status_details: dict[str, Literal['success', 'logicx_error']] = dict()
        if results_queue.empty():
            this_status = 'system_kill'
            this_error = 'subprocess exited without putting result (likely OOM)'
        else:
            this_status, this_error = results_queue.get()
            if this_status == 'success':
                for filepath in cvt_cache_dirpath.rglob('*.onnx'):
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(filepath)))
                        instances.append(instance)
                        artifacts.append(str(filepath.relative_to(cvt_cache_dirpath)))
                        this_status_details[str(filepath)] = 'success'
                    except Exception as exception:
                        this_status_details[str(filepath)] = 'logicx_error'

        if this_status == 'access_deny':
            # jump out of the opset loop if Repository can not found
            logger.warning(f'[opset {onnx_opset_version}] {this_status}: {this_error}')
            status[onnx_opset_version] = (this_status, this_status_details)
            # A 404 is a repository-level error rather than an opset-specific failure.
            # Stop here, so `status` may not contain entries for subsequent opsets.
            # Keep this semantic difference in mind if this logic is changed in the future.
            break
            
        if this_status == 'convert_error':
            # Permanent error -> skip remaining opsets entirely
            if is_permanent_error(this_error):
                logger.warning(f'[opset {onnx_opset_version}] {this_status}: {this_error}')
                status[onnx_opset_version] = (this_status, this_status_details)
                break

            # Unsupported operator with known minimum version -> jump there
            target_opset = get_minimum_opset_from_error(this_error)
            if target_opset is not None and target_opset > onnx_opset_version:
                logger.warning(f'[opset {onnx_opset_version}] {this_status}: {this_error}')
                status[onnx_opset_version] = (this_status, this_status_details)
                for i in range(opset_index + 1, len(opset_versions)):
                    if opset_versions[i] >= target_opset:
                        opset_index = i
                        break
                else:
                    # target_opset beyond our range -> exit
                    break
                continue
        # Log and record status
        if this_status == 'success':
            logger.info(f'[opset {onnx_opset_version}] {this_status}: {len(this_status_details)} artifacts created')
        elif this_error:
            logger.warning(f'[opset {onnx_opset_version}] {this_status}: {this_error}')
        status[onnx_opset_version] = (this_status, this_status_details)
        opset_index += 1

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_sb3_export(model_id: str, sb3_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue):
    import torch
    import gymnasium
    import stable_baselines3
    import sys

    class PureMathWrapper(torch.nn.Module):
        #Bypasses the strict probability distribution check in PyTorch 2.0+ Dynamo 
        #by manually mapping the pure mathematical tensor flow.
        def __init__(self, core_net, algo, is_dict, keys, is_discrete):
            super().__init__()
            self.core_net = core_net
            self.algo = algo
            self.is_dict = is_dict
            self.keys = keys
            self.is_discrete = is_discrete

        def forward(self, *args):
            if self.is_dict:
                obs = {k: v for k, v in zip(self.keys, args)}
            else:
                obs = args[0]
                
            if self.algo in ["DQN", "QRDQN"]:
                return self.core_net(obs)
                
            elif self.algo in ["SAC", "TD3", "DDPG", "TQC"]:
                features = self.core_net.extract_features(obs, self.core_net.features_extractor)
                latent_pi = self.core_net.latent_pi(features)
                mean_actions = self.core_net.mu(latent_pi)
                if self.algo in ["SAC", "TQC"]:
                    return torch.tanh(mean_actions)
                return mean_actions
                
            else: # PPO, A2C
                features = self.core_net.extract_features(obs)
                if self.core_net.share_features_extractor:
                    latent_pi, _ = self.core_net.mlp_extractor(features)
                else:
                    pi_features, _ = features
                    latent_pi = self.core_net.mlp_extractor.forward_actor(pi_features)
                    
                mean_actions = self.core_net.action_net(latent_pi)
                if self.is_discrete:
                    return torch.argmax(mean_actions, dim=1)
                return mean_actions

    sb3_algorithm_names = set(['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'])
    def infer_sb3_algorithm(model_id: str, sb3_model_path: pathlib.Path):
        for name in sb3_algorithm_names:
            if name in sb3_model_path.name.lower(): return name
        for name in sb3_algorithm_names:
            if name in model_id.lower(): return name
        return None

    def load_sb3_model(sb3_model_path: pathlib.Path, sb3_algorithm_name: str | None):
        custom_objects = {"optimize_memory_usage": False, "handle_timeout_termination": False}
        if sb3_algorithm_name is None:
            for name in sb3_algorithm_names:
                try:
                    return stable_baselines3.__dict__[name.upper()].load(sb3_model_path, device='cpu', custom_objects=custom_objects)
                except Exception:
                    continue
            raise ValueError(f"Could not load {sb3_model_path}.")
        else:
            return stable_baselines3.__dict__[sb3_algorithm_name.upper()].load(sb3_model_path, device='cpu', custom_objects=custom_objects)

    try:
        algo = infer_sb3_algorithm(model_id, sb3_model_path)
        model = load_sb3_model(sb3_model_path, algo)
        
        algo_name = model.__class__.__name__
        if 'recurrent' in algo_name.lower() or 'maskable' in algo_name.lower():
            raise NotImplementedError(f'Unsupported SB3 family: {algo_name}')

        if algo_name.upper() in {'SAC', 'TD3', 'DDPG', 'TQC'}:
            core_net = model.policy.actor
        elif algo_name.upper() in {'DQN', 'QRDQN'}:
            core_net = model.policy.q_net
        else:
            core_net = model.policy

        is_dict = isinstance(model.observation_space, gymnasium.spaces.Dict)
        keys = list(model.observation_space.spaces.keys()) if is_dict else []
        is_discrete = isinstance(model.action_space, gymnasium.spaces.Discrete)

        print(f"\nAnalyzing {algo_name} policy\n", file=sys.stderr)
        
        wrapped_model = PureMathWrapper(core_net, algo_name.upper(), is_dict, keys, is_discrete)
        wrapped_model.eval()

        model_device = next(wrapped_model.parameters()).device

        if is_dict:
            dummy_input = tuple(torch.randn(1, *model.observation_space.spaces[k].shape).to(model_device) for k in keys)
            input_names = [f'input_{k}' for k in keys]
        else:
            dummy_input = (torch.randn(1, *model.observation_space.shape).to(model_device), )
            input_names = ['input_observation']

        torch.onnx.export(wrapped_model, dummy_input, onnx_model_path, opset_version=onnx_opset_version, input_names=input_names, output_names=['output_action'])
        this_status = 'success'
        
    except Exception as exception:
        this_status = 'convert_error'
        import traceback
        print(f"\nFail. {model_id} Caused by：\n{traceback.format_exc()}\n", file=sys.stderr)
    finally:
        results_queue.put(this_status)

def convert_sb3(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None) -> tuple[dict[str, dict[int, Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance], list[str]]:
    status: dict[str, dict[int, Literal['success', 'oversize', 'access_deny', 'convert_error', 'system_kill', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    artifacts: list[str] = list()

    remote_sb3_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.zip'])
    from torch.onnx import _constants
    opset_versions = [v for v in get_onnx_opset_versions() if 14 <= v and v <= _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET]

    for remote_sb3_model_path in remote_sb3_model_paths:
        remote_sb3_model_name = os.path.splitext(remote_sb3_model_path)[0]

        try:
            sb3_model_path = pathlib.Path(hf_hub_download(model_id, remote_sb3_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            status[remote_sb3_model_name] = 'access_deny'
            continue

        status[remote_sb3_model_name] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(sb3_model_path))}.onnx')
        for onnx_opset_version in opset_versions:
            results_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_sb3_export, args=(model_id, sb3_model_path, onnx_model_path, onnx_opset_version, results_queue))
            subprocess.start()
            subprocess.join()

            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(pathlib.Path(onnx_model_path))))
                        instances.append(instance)
                        artifacts.append(f'{remote_sb3_model_name}')
                    except Exception:
                        this_status = 'logicx_error'

            status[remote_sb3_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_keras_export(keras_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue):
    from .miscs import tf2onnx_main_export

    if keras_model_path.is_dir():
        model_type = 'saved_model'
    if keras_model_path.is_file():
        model_type = 'keras'

    try:
        tf2onnx_main_export(keras_model_path, onnx_model_path, onnx_opset_version, model_type=model_type)
        this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_keras(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None) -> tuple[dict[str, dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance], list[str]]:
    status: dict[str, Literal['access_deny'] | dict[int, Literal['success', 'convert_error', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    artifacts: list[str] = list()
    remote_keras_model_paths = list()
    for remote_keras_model_path in get_huggingface_hub_model_siblings(model_id, suffixes=['.keras', '.hdf5', '.h5', '.pbtxt', '.pb']):
        if remote_keras_model_path.endswith('.pbtxt') or remote_keras_model_path.endswith('.pb'):
            remote_keras_model_paths.append((os.path.dirname(remote_keras_model_path), 'D'))
        else:
            remote_keras_model_paths.append((remote_keras_model_path, 'F'))
    remote_keras_model_paths = list(set(remote_keras_model_paths))

    for remote_keras_model_path, path_type in remote_keras_model_paths:
        remote_keras_model_name = os.path.splitext(remote_keras_model_path)[0]
        try:
            if path_type == 'D':
                allowed_pattern = '*' if remote_keras_model_path == '' else f'{remote_keras_model_path}/*'
                keras_model_path = pathlib.Path(snapshot_download(model_id, allow_patterns=allowed_pattern, cache_dir=ofc_cache_dirpath)).joinpath(remote_keras_model_path)
            if path_type == 'F':
                keras_model_path = pathlib.Path(hf_hub_download(model_id, remote_keras_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            status[remote_keras_model_name] = 'access_deny'
            continue

        status[remote_keras_model_name] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(keras_model_path))}.onnx')
        for onnx_opset_version in get_onnx_opset_versions():
            # tf2onnx only support 14 - 18 opset version
            if onnx_opset_version < 14 or 18 < onnx_opset_version:
                continue
            results_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_keras_export, args=(keras_model_path, onnx_model_path, onnx_opset_version, results_queue))
            subprocess.start()
            subprocess.join()

            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(onnx_model_path)))
                        instances.append(instance)
                        artifacts.append(f'{remote_keras_model_path}')
                    except Exception as exception:
                        this_status = 'logicx_error'
            status[remote_keras_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_tflite_export(tflite_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version: int, results_queue: multiprocessing.Queue):
    from .miscs import tf2onnx_main_export

    try:
        tf2onnx_main_export(tflite_model_path, onnx_model_path, onnx_opset_version, model_type='tflite')
        this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_tflite(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None) -> tuple[dict[str, dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]], list[Instance], list[str]]:
    status: dict[str, Literal['access_deny'] | dict[int, Literal['success', 'convert_error', 'system_kill', 'logicx_error']]] = dict()
    instances: list[Instance] = list()
    artifacts: list[str] = list()
    remote_tflite_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.tflite'])
    for remote_tflite_model_path in remote_tflite_model_paths:
        remote_tflite_model_name = os.path.splitext(remote_tflite_model_path)[0]
        try:
            tflite_model_path = pathlib.Path(hf_hub_download(model_id, remote_tflite_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            status[remote_tflite_model_name] = 'access_deny'
            continue

        status[remote_tflite_model_name] = dict()
        onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(tflite_model_path))}.onnx')
        for onnx_opset_version in get_onnx_opset_versions():
            # tf2onnx only support 14 - 18 opset version
            if onnx_opset_version < 14 or 18 < onnx_opset_version:
                continue
            results_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_tflite_export, args=(tflite_model_path, onnx_model_path, onnx_opset_version, results_queue))
            subprocess.start()
            subprocess.join()
            if results_queue.empty():
                this_status = 'system_kill'
            else:
                this_status = results_queue.get()
                if this_status == 'success':
                    try:
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(onnx_model_path)))
                        instances.append(instance)
                        artifacts.append(f'{remote_tflite_model_name}')
                    except Exception as exception:
                        this_status = 'logicx_error'

            status[remote_tflite_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def safe_onnx_export(origin_version_onnx_model_path: pathlib.Path, onnx_model_path: pathlib.Path, onnx_opset_version, results_queue: multiprocessing.Queue):
    try:
        origin_version_onnx_model = load_model(origin_version_onnx_model_path)
        onnx.save_model(onnx.version_converter.convert_version(origin_version_onnx_model, onnx_opset_version), onnx_model_path)
        this_status = 'success'
    except Exception as exception:
        this_status = 'convert_error'

    results_queue.put(this_status)


def convert_onnx(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', library_name: str | None = None) -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance], list[str]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()
    artifacts: list[str] = list()
    remote_onnx_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.onnx'])
    for remote_onnx_model_path in remote_onnx_model_paths:
        remote_onnx_model_name = os.path.splitext(remote_onnx_model_path)[0]
        try:
            onnx_model_path = pathlib.Path(hf_hub_download(model_id, remote_onnx_model_path, cache_dir=ofc_cache_dirpath))
        except Exception as exception:
            status[remote_onnx_model_name] = 'access_deny'
            continue

        status[remote_onnx_model_name] = dict()

        try:
            onnx_model = load_model(onnx_model_path)
        except Exception as exception:
            status[remote_onnx_model_name] = 'onnx_load_error'
            continue

        try:
            onnx_model_opset_version = get_onnx_model_opset_version(onnx_model)
        except Exception as exception:
            status[remote_onnx_model_name] = 'onnx_opset_error'
            continue

        for onnx_opset_version in get_onnx_opset_versions():
            if onnx_opset_version == onnx_model_opset_version:
                this_status = 'success'
                try:
                    instance = Instance()
                    instance.setup_logicx(convert(onnx_model))
                    instances.append(instance)
                    artifacts.append(f'{remote_onnx_model_name}')
                except Exception as exception:
                    this_status = 'logicx_error'
            else:
                other_version_onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(str(onnx_model_path))}.onnx')
                results_queue = multiprocessing.Queue()
                subprocess = multiprocessing.Process(target=safe_onnx_export, args=(onnx_model_path, other_version_onnx_model_path, onnx_opset_version, results_queue))
                subprocess.start()
                subprocess.join()
                if results_queue.empty():
                    this_status = 'system_kill'
                else:
                    this_status = results_queue.get()
                    if this_status == 'success':
                        try:
                            instance = Instance()
                            instance.setup_logicx(convert(load_model(other_version_onnx_model_path)))
                            instances.append(instance)
                            artifacts.append(f'{remote_onnx_model_name}')
                        except Exception as exception:
                            this_status = 'logicx_error'
            status[remote_onnx_model_name][onnx_opset_version] = this_status

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances, artifacts


def get_model_infos_and_convert_method(model_infos_filepath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']) -> tuple[list[dict[str, Any]], Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda']], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance], list[str]]]]:
    supported_frameworks: list[str] = ['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']
    supported_convert_methods: dict[str, Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda'], str | None], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance], list[str]]]] = dict(
        optimum=convert_optimum,
        onnx=convert_onnx,
        keras=convert_keras,
        tflite=convert_tflite,
        stable_baselines3=convert_sb3,
    )

    def get_model_frameworks(model_frameworks: list[Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']]) -> list[Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3']]:
        candidate_frameworks = set(model_frameworks) & set(supported_frameworks)
        model_frameworks = list()
        for supported_framework in supported_frameworks:
            if supported_framework in candidate_frameworks:
                model_frameworks.append(supported_framework)
        return model_frameworks

    model_infos: list[dict[str, Any]] = list()
    for model_info in load_json(model_infos_filepath):
        model_frameworks = get_model_frameworks(infer_supported_frameworks(model_info))
        if "stable-baselines3" in model_info.get("tags", []) and "stable_baselines3" not in model_frameworks:
            model_frameworks.append("stable_baselines3")
        
        if framework in model_frameworks:
            model_infos.append(model_info)

    convert_method = supported_convert_methods[framework]
    return model_infos, convert_method


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], model_size_limit: tuple[int, int]) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    specific_status_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    if specific_status_filepath.is_file():
        with open(specific_status_filepath, 'r') as specific_status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in specific_status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id

def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'], model_size_limit: tuple[int, int], convert_status: dict[str, dict[str, Any]], model_id: str):
    convert_status_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


def get_real_model_size(api: HfApi, model_id: str) -> int:
    import time
    from huggingface_hub.utils import HfHubHTTPError
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            live_info = api.model_info(model_id, files_metadata=True)
            total_size = 0
            for file in live_info.siblings:
                if file.size is not None:
                    total_size += file.size
            return total_size
        except HfHubHTTPError as e:
            if getattr(e.response, 'status_code', None) == 429:
                logger.warning(f"  [Size Check] {model_id} 429 Too Many Requests, sleep 5s...")
                time.sleep(5)
            elif getattr(e.response, 'status_code', None) in (401, 403, 404):
                logger.warning(f"  [Size Check] {model_id} Access denied/Not found ({e.response.status_code}).")
                return 999 * 1024 * 1024 * 1024 
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
            
    logger.error(f"  [Size Check] Failed to get real size for {model_id} after {max_retries} attempts.")
    return 999 * 1024 * 1024 * 1024 


def main(
    model_infos_filepath: pathlib.Path,
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path,
    device: Literal['cpu', 'cuda'] = 'cpu',
    framework: Literal['optimum', 'onnx', 'keras', 'tflite', 'stable_baselines3'] = 'optimum',
    model_size_limit_l: int | None = None,
    model_size_limit_r: int | None = None,
    token: str | None = None,
    estimate: bool = False,
):
    """
    Retrieve Metadata of HuggingFace Models and Save Them Into Files.

    :param model_ids_filepath: _description_
    :type model_ids_filepath: pathlib.Path
    :param save_dirpath: _description_
    :type save_dirpath: pathlib.Path
    :param cache_dirpath: _description_
    :type cache_dirpath: pathlib.Path
    :param device: _description_, defaults to 'cpu'
    :type device: Literal[&#39;cpu&#39;, &#39;cuda&#39;], optional
    :param framework: _description_, defaults to 'optimum'
    :type framework: Literal[&#39;optimum&#39;, &#39;onnx&#39;, &#39;keras&#39;, &#39;tflite&#39;], optional
    :param model_size_limit_l: _description_, defaults to None
    :type model_size_limit_l: int | None, optional
    :param model_size_limit_r: _description_, defaults to None
    :type model_size_limit_r: int | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    In this project we have a concept called Origin. Origin is a tuple of (hub, owner, name).

    HuggingFace Hub is a place where people can share their models, datasets, and scripts.
    This project hardcodes the hub as 'HuggingFace'.
    The Naming Convention of the Mdoel ID on HuggingFace Hub follows the format: {owner}/{name}.
    Thus the Origin of a Implementation, often called as Model which is a instance of a LogicX a.k.a. Neural Network Architecture (NNA), on HuggingFace Hub is Origin('HuggingFace', owner, name).

    Model Infos are retrieved from the HuggingFace Hub and sorted with lastModified time in descending order, and saved into a JSON file by using command `younger-logics-ir create onnx retrieve huggingface --mode Model_Infos --save-dirpath ${SAVE_DIRPATH}`.

    .. note::
        The Instances are saved into the directory named as 'Instances-HuggingFace-{Framework}' under the save_dirpath.

    """

    model_size_limit_l = model_size_limit_l or 0
    logger.info(f'   Model Size Left Limit: {get_human_readable_size_representation(model_size_limit_l)}.')

    model_size_limit_r = model_size_limit_r or 1024 * 1024 * 1024 * 1024 * 1024
    logger.info(f'   Model Size Right Limit: {get_human_readable_size_representation(model_size_limit_r)}.')

    model_size_limit = (model_size_limit_l, model_size_limit_r)

    model_infos, convert_method = get_model_infos_and_convert_method(model_infos_filepath, framework)
    if estimate:
        logger.info(f'Only Estimate. Models To Be Converted: {len(model_infos)}; Model Infos Filename: {model_infos_filepath.name}.')
        return

    # Instances
    instances_dirpath = save_dirpath.joinpath(f'Instances')
    create_dir(instances_dirpath)

    # READMES
    readmes_dirpath = save_dirpath.joinpath(f'READMES')
    create_dir(readmes_dirpath)

    # Official
    ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFOfc')
    create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFCvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFSts')
    create_dir(sts_cache_dirpath)
    
    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit)
    number_of_converted_models = len(convert_status)
    logger.info(f'-> Previous Converted Models: {number_of_converted_models}')

    if token is not None:
        logger.info(f'-> HuggingFace Token Provided. Now Logging In ...')
        login(token)
    else:
        logger.info(f'-> HuggingFace Token Not Provided. Now Accessing Without Token ...')

    hf_api = HfApi(token=token)
    oversized_models_filepath = save_dirpath.joinpath(f'skipped_size_models_{framework}.jsonl')

    logger.info(f'-> Instances Creating ...')
    with tqdm.tqdm(total=len(model_infos), desc='Create Instances') as progress_bar:
        for convert_index, model_info in enumerate(model_infos, start=1):
            model_id = model_info['id']
            
            used_storage = get_real_model_size(hf_api, model_id)
            
            if used_storage < model_size_limit_l or used_storage > model_size_limit_r:
                real_save_size = used_storage if used_storage < 900 * 1024 * 1024 * 1024 else -1
                
                #return unknown if meet 404 error
                display_size = get_human_readable_size_representation(real_save_size) if real_save_size >= 0 else "Unknown"
                logger.warning(f"-> Skip! Model {model_id} real size ({display_size}) is out of limit bounds [{get_human_readable_size_representation(model_size_limit_l)}, {get_human_readable_size_representation(model_size_limit_r)}].")
                
                model_info['usedStorage_real'] = real_save_size 
                with open(oversized_models_filepath, 'a', encoding='utf-8') as f:
                    f.write(saves_json(model_info) + '\n')
                
                progress_bar.set_description(f'Size-Limit, Skip - {model_id}')
                progress_bar.update(1)
                continue

            if framework == 'optimum':
                tags = set(model_info.get('tags', []))
                library_name = next(
                    (lib for tag, lib in [('sentence-transformers','sentence_transformers'),
                                          ('transformers','transformers'),
                                          ('diffusers','diffusers'),
                                          ('timm','timm')]
                     if tag in tags),
                    None
                )
            else:
                library_name = None
                
            if last_handled_model_id is not None:
                if model_id == last_handled_model_id:
                    last_handled_model_id = None
                progress_bar.set_description(f'Converted, Skip - {model_id}')
                progress_bar.update(1)
                continue

            status, instances, artifacts = convert_method(model_id, cvt_cache_dirpath, ofc_cache_dirpath, device, library_name=library_name)

            model_owner, model_name = model_id.split('/')
            for index, (instance, artifact) in enumerate(zip(instances, artifacts), start=1):
                instance.insert_label(
                    Implementation(
                        origin=Origin(YLIROriginHub.HUGGINGFACE, model_owner, model_name, artifact),
                        like=model_info['likes'],
                        download=model_info['downloadsAllTime'],
                    )
                )
                instance_unique = instance.unique
                try:
                    instance.save(instances_dirpath.joinpath(instance_unique))
                except FileExistsError as exception:
                    logger.warning(f'-> Skip! Instance Already Exists: {instance_unique}')

            try:
                readme = get_huggingface_hub_model_readme(model_id, token=token)
            except Exception as exception:
                readme = ''

            if readme == '':
                pass
            else:
                save_json(readme, readmes_dirpath.joinpath(f'{model_owner}_YLIR_{model_name}.json'))

            
            set_convert_status_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit, status, model_id)
            clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)

            progress_bar.set_description(f'Convert - {model_id}')
            progress_bar.update(1)

    logger.info(f'-> Instances Created.')