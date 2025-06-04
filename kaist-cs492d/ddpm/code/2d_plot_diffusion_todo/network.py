import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # hidden_size : mlp로 변형시킬 크기, TimeLinear의 __init__에서 dim_out 값 전달 
        # frequency_embedding_size : TimeLinear에서는 별도의 입력값 x, default인 256을 사용함  

        self.mlp = nn.Sequential( # 두번의 linear와 한범의 SiLU 통과 
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        단일 숫자 시간 정보 t를 주어진 차원 수(dim)만큼의 sinusoidal embedding으로 변환해주는 함수.

        Args:
            t: (B,) 또는 (N,) shape의 텐서. 각 배치에 해당하는 시간 스텝 값 (정수 또는 소수 가능)
            dim: 최종 embedding 벡터의 차원 수 (보통 256). → output은 (B, dim)
            max_period: 주파수 감쇠의 최댓값. 낮은 주파수부터 높은 주파수까지 넓게 커버하는 데 사용됨.

        Returns:
            embedding: shape (B, dim)의 sinusoidal time embedding 텐서
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2 # 절반은 sin, 나머지 절반은 cos

        freqs = torch.exp( # 주파수를 작은것부터 긴것까지 logscale로 생성 
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] 
        # 주파수를 args로 만들어서 sin, cos 각각에 할당 
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        # timestep_embedding -> mlp 순서로 실행 
        # timestep_embedding의 dim으로 frequency_embedding_size 할당 
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeLinear(nn.Module): 
    # 모델에 시간 정보를 주입하는 간단한 모듈, 시간에 따라 선형 레이어의 출력이 달라지도록 설계됨
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int): 
        # dim_in, dim_out : 입력될때 feature의 차원과, 연산을 위해 차원 조정된 feature의 차원 
        # num_timesteps는 일단 사용하고 있지는 않다? 
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out) # dim_out 크기의 시간 정보 임베딩 벡터를 생성하는 모듈
        self.fc = nn.Linear(dim_in, dim_out)  # 입력 x를 시간 임베딩과 곱할 수 있도록 dim_out으로 projection

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: 입력 feature 텐서 (B, dim_in)
        # t: 시간 정보 텐서 (B,) 또는 (1,)
        x = self.fc(x) # (B, dim_out)로 변환
        alpha = self.time_embedding(t).view(-1, self.dim_out) # 시간 임베딩 결과 (B, dim_out)

        return alpha * x  # 시간에 따라 feature를 element-wise scaling


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        ######## TODO ########
        # DO NOT change the code outside this part.

        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # 현재 시간 t의 X_t를 받아서 노이즈를 에측
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        ######## TODO ########
        
        # x_t에 포함된 노이즈 패턴 확인
        # t가 어느정도인지 확인
        # MLP로 이 둘을 입력으로 해서 결과를 출력 

        ######################
        return x
