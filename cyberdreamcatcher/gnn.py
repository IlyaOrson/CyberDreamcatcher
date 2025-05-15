from typing import Optional, Tuple, Union

from rich.console import Console
from rich import inspect as rin

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

# NOTE custom version to avoid automatic template generation
# from torch_geometric.nn.conv import MessagePassing
from cyberdreamcatcher.message_passing import NoTemplateMessagePassing


# class GATGlobalConv(MessagePassing):
class GATGlobalConv(NoTemplateMessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper,

    + plus a global node channel as in the TacticsAI paper,
    https://doi.org/10.1038/s41467-024-45965-x.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        self.lin_l = Linear(
            in_channels,
            heads * out_channels,
            bias=bias,
            weight_initializer="glorot",
        )
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
        else:
            self.lin_edge = None

        if global_dim is not None:
            self.lin_glob = Linear(
                global_dim,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
        else:
            self.lin_glob = None

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_glob is not None:
            self.lin_glob.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        global_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            global_attr (torch.Tensor): The global graph features.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            global_attr=global_attr,
        )

        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(
            return_attention_weights, bool
        ):  # TODO: use attention weights for explainability
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        global_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_latent = self.lin_edge(edge_attr)
            edge_latent = edge_latent.view(-1, self.heads, self.out_channels)
            x = x + edge_latent

        # NOTE sum/product broadcasts over incompatible dimensions and
        # its done per element over matching dimensions
        if global_attr is not None:
            assert self.lin_glob is not None
            global_latent = self.lin_glob(global_attr)
            global_latent = global_latent.view(-1, self.heads, self.out_channels)
            x = x + global_latent

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        # console = Console(record=True)
        # rin(self, console=console)
        # return console.export_text(styles=True)

        console = Console()
        with console.capture() as capture:
            rin(self)
        return capture.get()
