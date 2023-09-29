from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import torch


class AbsEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class CoverEncoder(AbsEncoder):
    def __init__(self,input_size):
        super(CoverEncoder, self).__init__()
        self.outputsize = input_size

    def output_size(self) -> int:
        return self.outputsize

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return xs_pad,ilens

class AVOutEncoder(AbsEncoder):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

if __name__ == "__main__":
    class tryencoder(AVOutEncoder):
        def output_size(self) -> int:
            return 0
        def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            return 0
    
    tryencoder = tryencoder()
    if isinstance(tryencoder,AVOutEncoder) and isinstance(tryencoder,AbsEncoder):
        print("succed")


