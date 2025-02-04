import sys
from copy import copy
from collections import defaultdict, namedtuple  # , ChainMap
from itertools import combinations, product, repeat  # , starmap

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from bidict import bidict
from rich.pretty import pprint
import torch

from CybORG import CybORG
from CybORG.Shared.Enums import TrinaryEnum
from CybORG.Agents import RedMeanderAgent  # , TestAgent
from CybORG.Agents.Wrappers import ChallengeWrapper  # , BaseWrapper

# NOTE not sure if this limits are actually enforced in CybORG
from CybORG.Shared.ActionSpace import MAX_PORTS
from CybORG.Shared.AgentInterface import MAX_CONNECTIONS

from torch import tensor
from torch_geometric.data import Data

from cyberdreamcatcher.utils import get_scenario, enumerate_bidict
from cyberdreamcatcher.plots import (
    plot_observation,
    plot_observation_encoded,
    plot_feasible_connections,
)


# NOTE does not comply with the gym observation space restrictions
# class GraphWrapper(gym.Env):
class GraphWrapper:
    agent_name = "Blue"

    HostProperties = namedtuple("Host", ("subnet", "num_local_ports", "malware"))

    host_embedding_size = 3
    edge_embedding_size = 1
    global_embedding_size = 3

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, scenario=None, max_steps=100, render_mode="human", verbose=False
    ) -> None:
        self.step_counter = None
        self.max_steps = max_steps
        self.verbose = verbose

        if not scenario:
            self.scenario_path = get_scenario(name="Scenario2", from_cyborg=True)
            self.scenario_name = "CybORG's Scenario2"
        else:
            self.scenario_path = get_scenario(name=scenario, from_cyborg=False)
            self.scenario_name = scenario

        self.cyborg = CybORG(self.scenario_path, "sim", agents={"Red": RedMeanderAgent})
        self.env_controller = self.cyborg.environment_controller
        self.scenario = self.env_controller.scenario

        self.host_names = self.scenario.hosts
        self.subnet_names = self.scenario.subnets
        self.action_names = self.scenario._scenario["Agents"][self.agent_name][
            "actions"
        ]
        self.global_actions_names = ("Sleep", "Monitor")

        # Form enumeration mappings
        self.subnet_enumeration = enumerate_bidict(self.subnet_names)
        self.host_enumeration = enumerate_bidict(self.host_names)
        self.action_enumeration = enumerate_bidict(self.action_names)

        # Handy properties
        self.num_subnets = len(self.subnet_names)
        self.num_hosts = len(self.host_names)
        self.num_actions = len(self.action_names)
        self.num_actions_per_node = self.num_actions - len(self.global_actions_names)

        # Extract possible actions from cyborg
        # action_space = self.env_controller.agent_interfaces[self.agent_name].action_space
        self.set_feasible_actions()

        # NOTE this depends on the random IPs assigned so need to be called after each environment reset as well
        # Initialize feasiable connection graph with the structure from the scenario
        self.set_feasible_connections()

        # NOTE  the "true" state is not really updated because the observations are updated instead
        #       directly in self.env_controller.observation["Blue"]
        #       in the self.env_controller.step(...) method
        # state = ec.state
        # ec_st_true = ec.get_true_state(ec.INFO_DICT["True"]).data
        # ec_obs_true = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["True"])).data
        # ec_obs_blue = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["Blue"]), "Blue").data

        # ChallengeWrapper > OpenAIGymWrapper > EnumActionWrapper > BlueTableWrapper > TrueTableWrapper > CyBORG
        self.challenge = ChallengeWrapper(
            agent_name=self.agent_name, env=self.cyborg, max_steps=self.max_steps
        )
        self.openai_gym = self.challenge.env
        self.enum_action = self.openai_gym.env
        self.blue_table = self.enum_action.env
        self.true_table = self.blue_table.env

        # This imitates the logic in BlueTable._process_initial_obs()
        self.blue_baseline = {
            k: v for k, v in self.get_raw_observation().items() if k != "success"
        }

        self.previous_action = None
        assert str(self.gym_to_cyborg_action([0, 0])) == "Sleep"
        # always "sleep" in the first move
        self.previous_action_encoding = torch.tensor([0, 0], dtype=torch.float)

        # Set gymnasium properties
        self.reward_range = (float("-inf"), float("inf"))
        self.action_space = gym.spaces.MultiDiscrete([self.num_hosts, self.num_actions])

        # NOTE  not very useful since unexpected connections appear regardless of layout constraints... and
        #       nested dict observations are not really supported by stable baselines 3, only super siple plain dicts
        # self.observation_space = self._build_dict_obs_space()
        self.observation_space = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            # plt.ion()
            # plt.ioff()
            self.fig, self.axis = plt.subplots(1, 2)
            self._node_positions = plot_feasible_connections(self)

    def _build_dict_obs_space(self):
        # host_properties[host] = [subnet, num_local_ports, malware]
        host_props = gym.spaces.Dict(
            {
                "host": gym.spaces.Discrete(self.num_hosts),
                "properties": gym.spaces.Dict(
                    {
                        "subnet": gym.spaces.Discrete(self.num_subnets),
                        "num_local_ports": gym.spaces.Discrete(MAX_PORTS),
                        "malware": gym.spaces.Discrete(2),  # boolean
                    }
                ),
            }
        )
        connection_props = gym.spaces.Dict(
            {
                "origin": gym.spaces.Discrete(self.num_hosts),
                "target": gym.spaces.Discrete(self.num_hosts),
                "connections": gym.spaces.Discrete(MAX_CONNECTIONS),
            }
        )
        observation_space = gym.spaces.Dict(
            {
                "hosts": gym.spaces.Tuple(repeat(host_props, self.num_hosts)),
                "connections": gym.spaces.Tuple(
                    repeat(connection_props, self.num_feasible_connections)
                ),
                "previous_action": self.action_space,
            }
        )
        return observation_space

    def set_feasible_connections(self):
        "Extract graph layout from State object in CybORG, which is populated from the Scenario config."

        # subnet name <--> ip network (subnet)
        self.subnet_cidr_map = bidict(self.env_controller.subnet_cidr_map)
        # hostname <--> ip address
        self.hostname_ip_map = bidict(self.env_controller.hostname_ip_map)
        # ip_address --> subnet
        # self.env_controller.state.get_subnet_containing_ip_address(ip_address)

        # feasible connections between hosts in different subnet
        self.internet_connections = []
        # self.env_controller.state.hosts --> dictionary from host names to Host() objects
        for hostname, host in self.env_controller.state.hosts.items():
            # self_reference = False
            # if interface appears here, suppose a connection is feasible
            for name, interface in host.info.items():
                if name == hostname:
                    # TODO should we add a self loop
                    # self_reference = True
                    continue  # NOTE does this flag a node available for red attacks?
                origin_remote = (hostname, name)
                self.internet_connections.append(origin_remote)

        # hostname --> ip network (subnet)
        self.hostname_subnet_map = {}
        # hostname -->  subnet name
        self.hostname_subnetname_map = {}
        # subnet name --> hostnames
        self.subnet_hostnames_map = {}

        # feasible connections between hosts in the same subnet
        self.intranet_connections = []
        subnets = self.env_controller.state.subnets.values()
        for subnet in subnets:
            subnet_name = subnet.name
            subnet_hostnames = []

            for ip in subnet.ip_addresses:
                hostname = self.hostname_ip_map.inv[ip]
                subnet_hostnames.append(hostname)
                self.hostname_subnet_map[hostname] = subnet.cidr
                self.hostname_subnetname_map[hostname] = subnet.name

            self.subnet_hostnames_map[subnet_name] = subnet_hostnames

            # add intranet feasible connections as all possible links between hosts in the same subnet
            for source, target in combinations(subnet_hostnames, 2):
                self.intranet_connections.append((source, target))
                self.intranet_connections.append((target, source))

        self.feasible_connections = (
            self.internet_connections + self.intranet_connections
        )
        self.feasible_connections_set = set(self.feasible_connections)
        self.num_feasible_connections = len(self.feasible_connections)
        self.connections_enumeration = enumerate_bidict(self.feasible_connections)

    def set_feasible_actions(self):
        """Iterate over the action classes reported by cyborg and instantiate each of them
        with the parameters available in the action space which match their signature.
        """

        global_signatures = [(action, None) for action in self.global_actions_names]

        host_actions = (
            action
            for action in self.action_names
            if action not in self.global_actions_names
        )
        action_host_signatures = product(host_actions, self.host_names)
        self.feasible_actions = list(action_host_signatures) + global_signatures

        # Equivalent to the logic in EnumActionWrapper.action_space_change(action_space_dict)
        # self.feasible_action_instances = list(starmap(self.instantiate_action, self.feasible_actions))

    def _get_action_class(self, action_name):
        "Retrieve the action class defined in CybORG from the action name."
        action_module = sys.modules["CybORG.Shared.Actions"]
        action_class = getattr(action_module, action_name)
        return action_class

    def instantiate_action(self, action_name, host_name):
        """Create instantiate the class object with the given host."""

        action_class = self._get_action_class(action_name)

        if action_name == "Sleep":
            action = action_class()
        elif action_name == "Monitor":
            action = action_class(session=0, agent=self.agent_name)
        else:
            action = action_class(session=0, agent=self.agent_name, hostname=host_name)
        return action

    def gym_to_cyborg_action(self, gym_action):
        "Converts gymnasium action to the equivalent cyborg action."
        host_idx, action_idx = gym_action
        action_name = self.action_names[action_idx]
        host_name = self.host_names[host_idx]
        if action_name in self.global_actions_names:
            host_name = None  # global actions ignore host selection
        assert (action_name, host_name) in self.feasible_actions
        action_instance = self.instantiate_action(action_name, host_name)
        return action_instance

    def distill_graph_observation(self, observation):
        """Extracts from the raw blue observation the information required
        to reconstruct the the blue table state but in a graph representation.
        """

        host_properties = {}
        connections_between_hosts = defaultdict(int)
        success_enum = None
        for host, properties in observation.items():
            if host == "success":
                success_enum = properties
                # NOTE The observation can be valuable even if the previous action failed
                # if properties.name == "FALSE":
                #     pass
                #     return self.host_properties_baseline, self.connections_baseline, success_enum
                continue

            num_local_ports = 0
            if "Processes" in properties:
                processes = properties["Processes"]

                local_ports_counter = defaultdict(int)
                remote_ports_counter = defaultdict(int)

                for process in processes:
                    if "Connections" in process:
                        assert len(process["Connections"]) == 1
                        connection = process["Connections"][0]

                        if "Transport Protocol" in connection:
                            continue  # FIXME double check

                        local_address = connection["local_address"]
                        remote_address = connection["remote_address"]
                        if local_address == remote_address:
                            # NOTE should self connections be included in the graph encoding?
                            if self.verbose:
                                print(
                                    f"Self-connection observed in {host}: {connection}"
                                )
                            # continue

                        local_host_name = self.hostname_ip_map.inv[local_address]
                        remote_host_name = self.hostname_ip_map.inv[remote_address]
                        local_remote_tuple = (local_host_name, remote_host_name)

                        assert host == local_host_name, "Utter nonsense again!"
                        # assert local_remote in self.feasible_connections, "Unfeasible connection appeared!"
                        if local_remote_tuple not in self.feasible_connections_set:
                            if self.verbose:
                                print(
                                    f"Unfeasible connection appeared! {local_host_name} --> {remote_host_name}"
                                )
                        connections_between_hosts[local_remote_tuple] += 1

                        local_port = connection["local_port"]
                        local_ports_counter[local_port] += 1

                        try:
                            remote_port = connection["remote_port"]
                            remote_ports_counter[remote_port] += 1
                        except KeyError:
                            if self.verbose:
                                print(
                                    f"Connection {local_remote_tuple} has no remote port!"
                                )

                num_local_ports = sum(local_ports_counter.values())

            malware = False
            if "Files" in properties:
                files = properties["Files"]
                malware = any(_file["Density"] >= 0.9 for _file in files)

            subnet_ip = self.hostname_subnet_map[host]
            subnet = self.subnet_cidr_map.inv[subnet_ip]
            # host_properties[host] = [subnet, num_local_ports, malware]
            host_properties[host] = self.HostProperties(
                subnet, num_local_ports, malware
            )

        if observation != self.blue_baseline:
            # extract processes per host
            anomalies = self.blue_table._detect_anomalies(observation)
            # flag if processes represent a connection or a file
            relevant_anomalies = {
                host: processes
                for host, processes in anomalies.items()
                if "Connections" in processes.keys() or "Files" in processes.keys()
            }
            if relevant_anomalies and self.verbose:
                pprint(relevant_anomalies)
                pprint(host_properties)
                pprint(connections_between_hosts)

        return host_properties, connections_between_hosts, success_enum

    def encode_graph_observation(
        self, host_properties, connections_between_hosts, success_enum
    ):
        """Transform the human understandable graph representation to a matrix encoding.
        Categorical values are not one-hot-encoded for now.
        """

        connections = copy(connections_between_hosts)

        node_matrix = np.zeros(
            (self.num_hosts, len(self.HostProperties._fields)), dtype="i"
        )  # int32
        for host_name in self.host_names:
            host_idx = self.host_enumeration[host_name]
            props = host_properties.get(
                host_name, self.host_properties_baseline[host_name]
            )
            subnet_id = self.subnet_enumeration[props.subnet]
            local_ports = props.num_local_ports
            malware_int = int(props.malware)
            node_matrix[host_idx, :] = (subnet_id, local_ports, malware_int)

        # This set difference needs to happen before any further access to the
        # connections object because it is a default dict and its keys change upon access
        unexpected_connections = connections.keys() - self.feasible_connections_set

        # load fixed layout connections
        edge_tuples = []
        edge_weights = []
        edge_index = np.zeros((2, self.num_feasible_connections), dtype="i")

        for source, target in self.feasible_connections:
            idx = self.connections_enumeration[(source, target)]

            source_id = self.host_enumeration[source]
            target_id = self.host_enumeration[target]
            tuple_id = (source_id, target_id)

            edge_index[:, idx] = tuple_id
            edge_tuples.append(tuple_id)

            current_connections = connections[(source, target)]
            edge_weights.append(current_connections)

        # append unfeasible connections found
        if unexpected_connections:
            extra_edge_tuples = []
            extra_edge_weights = []
            unexpected_edge_index = np.zeros(
                (2, len(unexpected_connections)), dtype="i"
            )

            for idx, (source, target) in enumerate(unexpected_connections):
                source_id = self.host_enumeration[source]
                target_id = self.host_enumeration[target]
                tuple_id = (source_id, target_id)

                unexpected_edge_index[:, idx] = tuple_id
                extra_edge_tuples.append(tuple_id)
                extra_edge_weights.append(connections[(source, target)])

            edge_tuples.extend(extra_edge_tuples)
            edge_weights.extend(extra_edge_weights)
            edge_index = np.hstack((edge_index, unexpected_edge_index))

        # edge weights are expected as a matrix of shape num_edges x num_attrs_per_edge
        edge_attr = np.array(edge_weights).reshape((-1, 1))

        if success_enum is None:
            success_value = TrinaryEnum.UNKNOWN.value
        else:
            success_value = success_enum.value
        prev_action_encoding = self.previous_action_encoding.clone().detach()
        success_encoding = torch.tensor([success_value], dtype=torch.float)

        return Data(
            x=tensor(node_matrix, dtype=torch.float),
            edge_index=tensor(edge_index, dtype=torch.long),
            edge_attr=tensor(edge_attr, dtype=torch.float),
            global_attr=torch.cat((prev_action_encoding, success_encoding)),
        )

    # def graph_to_gym_observation(self) TODO method to adapt graph to gymnasium space

    def reset(self, *, seed=None, options=None):
        self.step_counter = 0

        # CybORG does not expect options as a keyword
        if options:
            result = self.cyborg.reset(seed=seed, **options)
        else:
            result = self.cyborg.reset(seed=seed)

        self.set_feasible_connections()

        # Extract graph represention of blue the initial observation of the blue agent
        self.host_properties_baseline, self.connections_baseline, self.success_enum = (
            self.distill_graph_observation(self.blue_baseline)
        )
        observation = self.encode_graph_observation(
            self.host_properties_baseline,
            self.connections_baseline,
            self.success_enum,
        )

        # return np.array(result.observation), vars(result)

        graph_info = {
            "hosts": self.host_properties_baseline,
            "connections": self.connections_baseline,
        }
        # info = ChainMap(vars(result), graph_info)  # not supported by gymnasium wrappers
        info = vars(result)
        info.update(graph_info)
        return observation, info

    def step(self, action):
        action_instance = self.gym_to_cyborg_action(action)
        cyborg_result = self.cyborg.step(agent=self.agent_name, action=action_instance)

        # cyborg_observation = cyborg_result.observation
        host_properties, connections, success = self.get_graph_observation()
        observation = self.encode_graph_observation(
            host_properties, connections, success
        )

        reward = cyborg_result.reward
        terminated = cyborg_result.done

        truncated = False
        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True

        graph_info = {"hosts": host_properties, "connections": connections}
        # info = ChainMap(vars(cyborg_result), graph_info)
        info = vars(cyborg_result)
        info.update(graph_info)

        self.previous_action_encoding = action.float()
        self.previous_action = action_instance

        return observation, reward, terminated, truncated, info

    def render(self):
        # TODO add success status to plot
        host_properties, connections, success = self.get_graph_observation()
        observation = self.encode_graph_observation(
            host_properties, connections, success
        )
        if self.render_mode == "human":
            plot_observation(
                host_properties,
                connections,
                axis=self.axis[0],
                node_positions=self._node_positions,
                # show=True,
            )
            plot_observation_encoded(
                self,
                observation,
                node_positions=self._node_positions,
                axis=self.axis[1],
                # show=True,
            )
            if self.previous_action is None:
                self.fig.suptitle("Initial blue observation")
            else:
                self.fig.suptitle(f"Blue observation after {str(self.previous_action)}")
            self.fig.set_tight_layout(True)

    def get_encoded_observation(self):
        host_properties, connections, success = self.get_graph_observation()
        return self.encode_graph_observation(host_properties, connections, success)

    def get_graph_observation(self):
        raw_observation = self.get_raw_observation()
        return self.distill_graph_observation(raw_observation)

    def get_raw_observation(self):
        # NOTE with ec == CybORG.environment_controller
        # self.blue_table.get_observation("Blue") is equivalent to
        # ec.get_last_observation("Blue").data --> ec.observation["Blue"]
        return self.env_controller.observation[self.agent_name].data

    def get_true_table(self):
        return self.blue_table.get_table(output_mode="true_table")

    def get_blue_table(self):
        return self.blue_table.get_table(output_mode="blue_table")

    def get_last_action(self):
        return self.cyborg.get_last_action(self.agent_name)

    # NOTE use previous action in the graph repr with an independent linear transformation
    # def encode_last_action(self):
    #     """BlueTable logic relies only on the last action being of the broad type
    #     (Restore, Remove or Other)
    #     """
    #     action = self.get_last_action()
