import torch
import torch.nn as nn
import sys
import torchvision

class RecurrentHeighBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, height_map, proprioception):
        height_map = self.base_backbone(height_map)
        height_latent = self.combination_mlp(torch.cat((height_map, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        height_latent, self.hidden_states = self.rnn(height_latent[:, None, :], self.hidden_states)
        height_latent = self.output_mlp(height_latent.squeeze(1))
        
        return height_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


# def get_ScanEncoderBackbone(self, num_scan, scan_encoder_dims, activation=nn.ELU()):
#     scan_encoder = []
#     scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
#     scan_encoder.append(activation)
#     for l in range(len(scan_encoder_dims) - 1):
#         if l == len(scan_encoder_dims) - 2:
#             scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
#             scan_encoder.append(nn.Tanh())
#         else:
#             scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
#             scan_encoder.append(activation)
#     return nn.Sequential(*scan_encoder)



# class HeightOnlyFCBackbone(nn.Module):
#     def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
#         super().__init__()

#         self.num_frames = num_frames
#         activation = nn.ELU()
#         self.image_compression = nn.Sequential(
#             # [1, 58, 87]
#             nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
#             # [32, 54, 83]
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # [32, 27, 41]
#             activation,
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             activation,
#             nn.Flatten(),
#             # [32, 25, 39]
#             nn.Linear(64 * 25 * 39, 128),
#             activation,
#             nn.Linear(128, scandots_output_dim)
#         )

#         if output_activation == "tanh":
#             self.output_activation = nn.Tanh()
#         else:
#             self.output_activation = activation

#     def forward(self, images: torch.Tensor):
#         images_compressed = self.image_compression(images.unsqueeze(1))
#         latent = self.output_activation(images_compressed)

#         return latent