import torch 
class DimConvert(torch.nn.Module): #(B,T,D)->(B,T,D)
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
    ):  
        super().__init__()
        settings = {
            "in_channels":in_channels,
            "out_channels":out_channels,
            "kernel_size":1,
            "stride":1,
            "bias":False,
            "padding":0,
        }
        self.convert = torch.nn.Sequential(
            torch.nn.Conv1d(**settings),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.PReLU(out_channels),
            torch.nn.Dropout(0.1)
        )
    def forward(self,tensor):
        return self.convert(tensor.transpose(1,2)).transpose(1,2)

class NewDimConvert(torch.nn.Module): #(B,T,D)->(B,T,D)
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        dropout_rate: float = 0.1,
    ):  
        super().__init__()
        self.convert = torch.nn.Sequential(
            torch.nn.Linear(in_channels,out_channels),
            torch.nn.LayerNorm(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            )

    def forward(self,tensor):
        return self.convert(tensor)
