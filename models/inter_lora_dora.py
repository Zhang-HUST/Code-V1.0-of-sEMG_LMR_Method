import torch.nn as nn
import torch


# ---------------------Lora-------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.rand(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# ---------------------Dora-------------------
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        self.m = nn.Parameter(torch.ones(1, linear.out_features))

    def forward(self, x):
        linear_out = self.linear(x)
        lora_out = self.lora(x)
        lora_out_norm = lora_out / (lora_out.norm(p=2, dim=1, keepdim=True) + 1e-9)
        dora_modification = self.m * lora_out_norm
        return linear_out + dora_modification


# Freeze the linear layer
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)


# Replace the linear layer in the model with LinearWithLoRA
def convert_lora_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank=4, alpha=8))
        else:
            convert_lora_layers(module)


# Replace the linear layer in the model with LinearWithDoRA
def convert_dora_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithDoRA(module, rank=4, alpha=8))
        else:
            convert_lora_layers(module)
