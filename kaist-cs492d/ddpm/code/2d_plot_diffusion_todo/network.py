import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module): # 타임 임베딩 벡터를 만드는 클래스 
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


class TimeLinear(nn.Module): # 입력 피쳐를 선형변환하고, 타임 임베딩 벡터를 이용해서 스케일링 해주는 클래스 
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
    def __init__( # SimpleNet을 정의하는 메서드 
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int  # 히든 레이어들의 차원은 list로 관리 중 
    ):                                                                            # dim_in -> dim_hids -> ... -> dim_hids[-1] -> dim_out와 같이 연결
        super().__init__()
        """
        (TODO) Build a noise estimating network.  # 네트워크를 정의할때는 t는 빠지는데 foward시에 t가 주입된다. 

        Args: 
            dim_in: dimension of input               
            dim_out: dimension of output
            dim_hids: dimensions of hidden features  
            num_timesteps: number of timesteps
        """

        ######## TODO ########

        # __init__은 고정된 신경망을 정의하는 부분 : 모든 레이어를 TimeLinear로 감싸서 구성 

        # 입력값 : dim_in, dim_hids, dim_out, num_timesteps 을 활용하고 
        # 사용 도구들 : nn.Sequential등의 도구 

        # 모든 레이어를 담을 리스트 (nn.Sequential로 한 번에 처리할 경우)
        layers = []

        # 첫 번째 레이어: dim_in -> dim_hids[0]
        layers.append(TimeLinear(dim_in, dim_hids[0], num_timesteps))
        layers.append(nn.SiLU()) # 첫 레이어 뒤에 활성화 함수 추가

        # 중간 은닉 레이어들: dim_hids[i] -> dim_hids[i+1]
        for i in range(len(dim_hids) - 1):
            layers.append(TimeLinear(dim_hids[i], dim_hids[i+1], num_timesteps))
            layers.append(nn.SiLU()) # 중간 레이어 뒤에 활성화 함수 추가

        # 마지막 레이어: dim_hids[-1] -> dim_out
        layers.append(TimeLinear(dim_hids[-1], dim_out, num_timesteps))

        # 정의된 모든 레이어를 nn.Sequential로 묶어 하나의 네트워크로 만듦
        # 이렇게 하면 forward 메서드에서 x와 t를 순차적으로 전달하기 편리
        self.network = nn.Sequential(*layers)
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

        # 입력된 x를 먼저 TimeLinear forward에 통과 시켜야할거 같다. 
        # 그런데 여기서 TimeLinear 객체가 dim_in, dim_out, num_timestep만 입력으로 받는다. 
        # 이 부분에서 TimeLinear는 각각 hidden layer 앞에서 항상 통과 시켜줘야할 수도 있겠다. 

        
        ######################
        return x
