# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from pathlib import Path

EXPORT_POLICY = True

class PLAY_TYPE():
    TEACHER = 1
    VISION  = 2
    HEIGHT  = 3
    TEACHER_CONTROLLER = 4

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    args.headless = False
    args.debug = True

    args.task = "go1"
    args.device = 'cuda:0'

    args.exptid = "BCK-25"
    # args.checkpoint = 10_000
    play_type = PLAY_TYPE.TEACHER_CONTROLLER
    
    def update_args(args, play_type):
        if play_type in [PLAY_TYPE.TEACHER, PLAY_TYPE.TEACHER_CONTROLLER]:
            args.use_camera = False
            draw_heights = True
            draw_goals = True
            return args, draw_goals, draw_heights
        
        if play_type == PLAY_TYPE.VISION:
            args.exptid += "-student"
            args.use_camera = True
            # args.checkpoint = "200"
            draw_heights = False
            draw_goals = False
            return args, draw_goals, draw_heights
        
        if play_type == PLAY_TYPE.HEIGHT:
            args.exptid += "-height"
            args.distill_only_heading = True
            args.use_camera = False
            draw_heights = True
            draw_goals = False
            return args, draw_goals, draw_heights

    args, draw_goals, draw_heights = update_args(args, play_type)

    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "{}/logs/{}/".format(LEGGED_GYM_ROOT_DIR, args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.terrain.difficulty_scale = 1

    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    
    env_cfg.terrain.terrain_dict = {"smooth slope": 0.0, 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.0,
                                    "parkour_hurdle": 0.0,
                                    "parkour_flat": 0.0,
                                    "parkour_step": 0.0,
                                    "parkour_gap": 0.0, 
                                    "demo": 0.0,
                                    "parkour_backward": 1.00}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.play.draw_goals = draw_goals
    env_cfg.play.draw_heights = draw_heights

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    is_student = env.cfg.depth.use_camera or env.cfg.height.distill_only_heading

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    # if args.use_jit:
    #     path = os.path.join(log_pth, "traced")
    #     model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
    #     path = os.path.join(path, model)
    #     print("Loading jit for policy: ", path)
    #     policy_jit = torch.jit.load(path, map_location=env.device)
    # else:
    if is_student:
        policy = ppo_runner.get_student_actor_inference_policy(device=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if is_student:
        student_vision_encoder = ppo_runner.get_student_vision_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    infos['height'] = env.height_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_distill_heading else None


    if play_type == PLAY_TYPE.TEACHER_CONTROLLER:
        original_env_class = env.env_class[0]
        env_class_counter = 0
        target_yaw = 0
        joystick_control = True
    else:
        joystick_control = False

    if EXPORT_POLICY:
        print("Exporting policy...")
        import copy
        import onnxruntime as ort
        from functools import partial
        from torch import nn
        export_device = 'cpu'

        if is_student:
            torch_model_src = copy.deepcopy(ppo_runner.alg.student_actor).to(export_device)
        else:
            torch_model_src = copy.deepcopy(ppo_runner.alg.actor_critic).to(export_device)

        class ONNXExportWrapper(nn.Module):
            def __init__(self, model):
                super(ONNXExportWrapper, self).__init__()
                self.model = model

            def forward(self, observations):
                return self.model.act_inference(observations, hist_encoding=True, scandots_latent=None)

        torch_model = ONNXExportWrapper(torch_model_src)
        torch_input = obs[1,None,:].to(export_device)
        torch_model(torch_input)

        savename = Path(f'{LEGGED_GYM_ROOT_DIR}/onnx/{args.exptid}.onnx')
        savename.parent.mkdir(parents=True, exist_ok=True)
        onnx_program = torch.onnx.export(
            torch_model, 
            torch_input,
            savename,
            opset_version=9,
            input_names=["input"],
            output_names=["action"])
        ort_sess = ort.InferenceSession(savename)
        outputs = ort_sess.run(None, {'input': torch_input.numpy()})[0]
        print("Exported policy to", savename)
        print("mean abs difference", (np.abs(outputs - torch_model(torch_input).detach().numpy())).mean())

    for i in range(10*int(env.max_episode_length)): 
        # if play_type == PLAY_TYPE.TEACHER_CONTROLLER:
        #     evt_lst = env.gym.query_viewer_action_events(env.viewer)
        #     for evt in evt_lst:
        #         print(evt.action)
        #         if evt.action == "QUIT" and evt.value > 0:
        #             sys.exit()
        #         # elif evt.action == "toggle_viewer_sync" and evt.value > 0:
        #         #     env.enable_viewer_sync = not env.enable_viewer_sync

        #         if evt.action == 'toggle_walk' and evt.value > 0:
        #             if env_class_counter % 2 == 0:
        #                 env.env_class[0] = 17
        #             else:
        #                 env.env_class[0] = original_env_class
        #             env_class_counter += 1

        #         if evt.action == "vx_plus" and evt.value > 0:
        #             env.commands[0, 0] += 0.1
        #             env.commands[0, 0] = torch.clip(env.commands[0, 0], 0, 1)
        #         if evt.action == "vx_minus" and evt.value > 0:
        #             env.commands[0, 0] -= 0.1
        #             env.commands[0, 0] = torch.clip(env.commands[0, 0], 0, 1)
        #         if evt.action == "left_turn" and evt.value > 0:
        #             target_yaw += float(np.pi/12)
        #         if evt.action == "right_turn" and evt.value > 0:
        #             target_yaw -= float(np.pi/12)
        #         if evt.action == "pause" and evt.value > 0:
        #             env.pause = True
        #             while env.pause:
        #                 sleep(0.1)
        #                 env.gym.draw_viewer(env.viewer, env.sim, True)
        #                 for evt in env.gym.query_viewer_action_events(env.viewer):
        #                     if evt.action == "pause" and evt.value > 0:
        #                         env.pause = False
        #                 if env.gym.query_viewer_has_closed(env.viewer):
        #                     sys.exit()
        #     env.gym.poll_viewer_events(env.viewer)
        #     env.yaw_overwrite = target_yaw
            # print(env.target_yaw)

        if is_student:
            if infos["depth"] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = student_vision_encoder(infos["depth"], obs_student)
                vision_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            if infos["height"] != None:
                obs_prop = obs[:, :env.cfg.env.n_proprio].clone()
                obs_prop[:, 6:8] = 0
                height_latent_and_yaw = student_vision_encoder(infos["height"].clone(), obs_prop)  # clone is crucial to avoid in-place operation
                vision_latent = height_latent_and_yaw[:, :-2]
                yaw = 1.5*height_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5*yaw
        else:
            vision_latent = None
        
        # if hasattr(ppo_runner.alg, "student_actor"):
        #     actions = ppo_runner.alg.student_actor(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
        # else:
        actions = policy(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
        obs, _, rews, dones, infos = env.step(actions.detach(), joystick_control)
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        # print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
        #       "cmd vx", env.commands[env.lookat_id, 0].item(),
        #       "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        

if __name__ == '__main__':
    args = get_args()
    play(args)
