import torch.nn as nn


class DQN(nn.Module):
    """
    Code adjusted from the DQN model in the book:
        `Lapan, Maxim. 2020. Deep Reinforcement Learning Hands-On:
        Apply Modern RL Methods to Practical Problems of Chatbots,
        Robotics, Discrete Optimization, Web Automation, and More.
        Packt Publishing Ltd. p.145-146`
    """

    def __init__(self, input_shape, out_channels):
        super().__init__()
        if input_shape[1] != input_shape[2]:
            raise Exception(
                "The input_shape[1] and input_shape[2] must be \
                    equal (i.e. the input images have to be a square)"
            )
        k_p_s = [
            (5, 1, 2),
            (4, 0, 2),
            (3, 0, 1),
        ]  # Kernel, Padding and Stride for each layer
        feature_map_size = self.conv_feature_map_size(input_shape[1], k_p_s)
        in_channels = input_shape[0]
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=k_p_s[0][0],
                stride=k_p_s[0][2],
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=k_p_s[1][0],
                stride=k_p_s[1][2],
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=k_p_s[2][0],
                stride=k_p_s[2][2],
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * feature_map_size**2, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )

    def forward(self, x):
        return self.conv_pipe(x)

    @staticmethod
    def conv_feature_map_size(input_dim: int, kps: list[tuple[int, int, int]]) -> int:
        feature_map_size = input_dim
        for k, p, s in kps:
            """
            Formula for calculating the output dimension of a convolutional layer
            If an input_dimension, kernel, stride, padding convolution doesn't
            perfectly match the convolution stops early (i.e. a floor division
            is used)
            """
            feature_map_size = (
                feature_map_size - k + 2 * p
            ) // s + 1  # Use floor division as a dimension has to be an integer
        return int(feature_map_size)
