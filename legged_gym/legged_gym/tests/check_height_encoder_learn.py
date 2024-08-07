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

class PLAY_TYPE():
    PLAY_TEACHER = 1
    PLAY_VISION  = 2
    PLAY_HEIGHT  = 3

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


    play_type = PLAY_TYPE.PLAY_HEIGHT
    args.checkpoint = "49500"

    def update_args(args, play_type):
        if play_type == PLAY_TYPE.PLAY_TEACHER:
            args.exptid = "MAY-02"
            args.use_camera = False
            draw_heights = True
            draw_goals = True
            return args, draw_goals, draw_heights
        
        if play_type == PLAY_TYPE.PLAY_VISION:
            args.exptid = "MAY-02-student"
            args.use_camera = True
            # args.checkpoint = "200"
            draw_heights = False
            draw_goals = False
            return args, draw_goals, draw_heights
        
        if play_type == PLAY_TYPE.PLAY_HEIGHT:
            args.exptid = "MAY-02-height"
            args.distill_only_heading = True
            args.use_camera = False
            draw_heights = True
            draw_goals = True
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
                                    "parkour_hurdle": 1.0,
                                    "parkour_flat": 0.0,
                                    "parkour_step": 0.0,
                                    "parkour_gap": 0.0, 
                                    "demo": 0.0}
    
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
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if is_student:
        student_vision_encoder = ppo_runner.get_student_vision_encoder_inference_policy(device=env.device)

    print(student_vision_encoder)

    # actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    # infos = {}
    # infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    # infos['height'] = env.height_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_distill_heading else None

    # for i in range(10*int(env.max_episode_length)):
    #     if args.use_jit:
    #         if is_student:
    #             if infos["depth"] is not None:
    #                 vision_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
    #                 actions, vision_latent = policy_jit(obs.detach(), True, infos["depth"], vision_latent)
    #             else:
    #                 depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
    #                 actions, vision_latent = policy_jit(obs.detach(), False, depth_buffer, vision_latent)
    #         else:
    #             obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
    #             actions = policy(obs_jit)
    #     else:
    #         if is_student:
    #             if infos["depth"] is not None:
    #                 obs_student = obs[:, :env.cfg.env.n_proprio].clone()
    #                 obs_student[:, 6:8] = 0
    #                 depth_latent_and_yaw = student_vision_encoder(infos["depth"], obs_student)
    #                 vision_latent = depth_latent_and_yaw[:, :-2]
    #                 yaw = depth_latent_and_yaw[:, -2:]
    #             if infos["height"] != None:
    #                 obs_prop = obs[:, :env.cfg.env.n_proprio].clone()
    #                 obs_prop[:, 6:8] = 0
    #                 height_latent_and_yaw = student_vision_encoder(infos["height"].clone(), obs_prop)  # clone is crucial to avoid in-place operation
    #                 vision_latent = height_latent_and_yaw[:, :-2]
    #                 yaw = 1.5*height_latent_and_yaw[:, -2:]
    #             obs[:, 6:8] = 1.5*yaw
    #         else:
    #             vision_latent = None
            
    #         if hasattr(ppo_runner.alg, "student_actor"):
    #             actions = ppo_runner.alg.student_actor(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
    #         else:
    #             actions = policy(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
            
    #     obs, _, rews, dones, infos = env.step(actions.detach())
    #     if args.web:
    #         web_viewer.render(fetch_results=True,
    #                     step_graphics=True,
    #                     render_all_camera_sensors=True,
    #                     wait_for_page_load=True)
    #     print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
    #           "cmd vx", env.commands[env.lookat_id, 0].item(),
    #           "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
    #     id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
