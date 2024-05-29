# NOTE  Patch to avoid automatic module generation from templates
#       used for torchscript optimization in pytorch geometric

# import os.path as osp
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Union,
)

from torch import Tensor

from torch_geometric.inspector import Inspector
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
# from torch_geometric.template import module_from_template

from torch_geometric.nn.conv import MessagePassing

FUSE_AGGRS = {"add", "sum", "mean", "min", "max"}
HookDict = OrderedDict[int, Callable]


# overkill but oh well
class BlindInspector(Inspector):
    @property
    def can_read_source(self):
        return False


class NoTemplateMessagePassing(MessagePassing):
    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "sum",
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
    ) -> None:
        # super().__init__()
        super(MessagePassing, self).__init__()  # torch module init

        if flow not in ["source_to_target", "target_to_source"]:
            raise ValueError(
                f"Expected 'flow' to be either 'source_to_target'"
                f" or 'target_to_source' (got '{flow}')"
            )

        # Cast `aggr` into a string representation for backward compatibility:
        self.aggr: Optional[Union[str, List[str]]]
        if aggr is None:
            self.aggr = None
        elif isinstance(aggr, (str, Aggregation)):
            self.aggr = str(aggr)
        elif isinstance(aggr, (tuple, list)):
            self.aggr = [str(x) for x in aggr]

        self.aggr_module = aggr_resolver(aggr, **(aggr_kwargs or {}))
        self.flow = flow
        self.node_dim = node_dim

        # Collect attribute names requested in message passing hooks:

        # NOTE hack to avoid automaitc module generation from templates
        # self.inspector = Inspector(self.__class__)
        self.inspector = BlindInspector(self.__class__)

        self.inspector.inspect_signature(self.message)
        self.inspector.inspect_signature(self.aggregate, exclude=[0, "aggr"])
        self.inspector.inspect_signature(self.message_and_aggregate, [0])
        self.inspector.inspect_signature(self.update, exclude=[0])
        self.inspector.inspect_signature(self.edge_update)

        self._user_args: List[str] = self.inspector.get_flat_param_names(
            ["message", "aggregate", "update"], exclude=self.special_args
        )
        self._fused_user_args: List[str] = self.inspector.get_flat_param_names(
            ["message_and_aggregate", "update"], exclude=self.special_args
        )
        self._edge_user_args: List[str] = self.inspector.get_param_names(
            "edge_update", exclude=self.special_args
        )

        # Support for "fused" message passing:
        self.fuse = self.inspector.implements("message_and_aggregate")
        if self.aggr is not None:
            self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS

        # Hooks:
        self._propagate_forward_pre_hooks: HookDict = OrderedDict()
        self._propagate_forward_hooks: HookDict = OrderedDict()
        self._message_forward_pre_hooks: HookDict = OrderedDict()
        self._message_forward_hooks: HookDict = OrderedDict()
        self._aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._aggregate_forward_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_hooks: HookDict = OrderedDict()
        self._edge_update_forward_pre_hooks: HookDict = OrderedDict()
        self._edge_update_forward_hooks: HookDict = OrderedDict()

        # root_dir = osp.dirname(osp.realpath(__file__))
        jinja_prefix = f"{self.__module__}_{self.__class__.__name__}"
        # # Optimize `propagate()` via `*.jinja` templates:
        # if not self.propagate.__module__.startswith(jinja_prefix):
        #     if self.inspector.can_read_source:
        #         module = module_from_template(
        #             module_name=f"{jinja_prefix}_propagate",
        #             template_path=osp.join(root_dir, "propagate.jinja"),
        #             tmp_dirname="message_passing",
        #             # Keyword arguments:
        #             module=self.__module__,
        #             collect_name="collect",
        #             signature=self._get_propagate_signature(),
        #             collect_param_dict=self.inspector.get_flat_param_dict(
        #                 ["message", "aggregate", "update"]
        #             ),
        #             message_args=self.inspector.get_param_names("message"),
        #             aggregate_args=self.inspector.get_param_names("aggregate"),
        #             message_and_aggregate_args=self.inspector.get_param_names(
        #                 "message_and_aggregate"
        #             ),
        #             update_args=self.inspector.get_param_names("update"),
        #             fuse=self.fuse,
        #         )

        #         # Cache to potentially disable later on:
        #         self.__class__._orig_propagate = self.__class__.propagate
        #         self.__class__._jinja_propagate = module.propagate

        #         self.__class__.propagate = module.propagate
        #         self.__class__.collect = module.collect
        #     else:
        #         self.__class__._orig_propagate = self.__class__.propagate
        #         self.__class__._jinja_propagate = self.__class__.propagate
        if not self.propagate.__module__.startswith(jinja_prefix):
            self.__class__._orig_propagate = self.__class__.propagate
            self.__class__._jinja_propagate = self.__class__.propagate

        # # Optimize `edge_updater()` via `*.jinja` templates (if implemented):
        # if (
        #     self.inspector.implements("edge_update")
        #     and not self.edge_updater.__module__.startswith(jinja_prefix)
        #     and self.inspector.can_read_source
        # ):
        #     module = module_from_template(
        #         module_name=f"{jinja_prefix}_edge_updater",
        #         template_path=osp.join(root_dir, "edge_updater.jinja"),
        #         tmp_dirname="message_passing",
        #         # Keyword arguments:
        #         module=self.__module__,
        #         collect_name="edge_collect",
        #         signature=self._get_edge_updater_signature(),
        #         collect_param_dict=self.inspector.get_param_dict("edge_update"),
        #     )

        #     self.__class__.edge_updater = module.edge_updater
        #     self.__class__.edge_collect = module.edge_collect

        # Explainability:
        self._explain: Optional[bool] = None
        self._edge_mask: Optional[Tensor] = None
        self._loop_mask: Optional[Tensor] = None
        self._apply_sigmoid: bool = True

        # Inference Decomposition:
        self.decomposed_layers = decomposed_layers
