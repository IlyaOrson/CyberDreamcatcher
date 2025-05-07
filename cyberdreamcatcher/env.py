from copy import copy
from pprint import pformat
from collections import defaultdict, namedtuple
from itertools import combinations, product

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from bidict import bidict
import torch
import logging

from CybORG import CybORG
from CybORG.Shared.Enums import TrinaryEnum
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import (
    TrueTableWrapper,
    BlueTableWrapper,
    RedTableWrapper,
)  # , ChallengeWrapper

from torch import tensor
from torch_geometric.data import Data

from cyberdreamcatcher.utils import (
    get_scenario,
    enumerate_bidict,
    instantiate_action,
    get_action_names,
)
from cyberdreamcatcher.plots import (
    plot_observation,
    plot_observation_encoded,
    plot_feasible_connections,
)

LOGGER = logging.getLogger(__name__)


# NOTE: override Wrappers basic methods to avoid calling cyborg.reset() (which regenerates IPs)
#       cyborg is managed directly in our environment instead of through wrappers (to be able to use both blue/red tables)


class TrueTable(TrueTableWrapper):
    def reset(self, cyborg_result):
        self.scanned_ips = set()
        self.step_counter = -1
        obs = cyborg_result.observation
        # do not rewrite cyborg_result.observation, only return the observation
        return self.observation_change(obs)


class BlueTable(BlueTableWrapper):
    def __init__(self, env=None, agent=None, output_mode="table"):
        self.env = TrueTable(env=env, agent=agent)
        # self.agent = agent  # not really used in BlueTableWrapper

        self.baseline = None
        self.output_mode = output_mode
        self.blue_info = {}

    def reset(self, cyborg_result):
        # calls TrueTable.reset() which avoids cyborg.reset()
        obs = self.env.reset(cyborg_result)

        self._process_initial_obs(obs)  # populates self.blue_info

        return self.observation_change(obs, baseline=True)

    # calling observation_change() is basically cyborg.step() + self.observation_change()
    # but these patches avoid calling cyborg.step() to manage this separately
    # def step(self, cyborg_result):
    #     obs = cyborg_result.observation
    #     return self.observation_change(obs)


class RedTable(RedTableWrapper):
    def __init__(self, env=None, agent=None, output_mode="table"):
        self.env = TrueTable(env=env, agent=agent)
        # self.agent = agent  # not really used in RedTableWrapper

        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.output_mode = output_mode
        self.success = None

    def reset(self, cyborg_result):
        # calls TrueTable.reset() which avoids cyborg.reset()
        obs = self.env.reset(cyborg_result)

        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.success = None

        return self.observation_change(obs)


# class GraphEnv(gym.Env):  # graph observation are not supported by gymnasium observation space restrictions
class GraphEnv:
    agent_name = "Blue"

    # FIXME should be 2 if an exploit connection will be flagged on the edge it appears, only flagged on the host for now
    host_encoding_dim = 5
    edge_encoding_dim = 1
    global_encoding_dim = 1

    HostObs = namedtuple("Host", ("num_local_ports", "exploit", "malware"))
    # port 4444 is hard-coded to represent exploited ports
    EdgeObs = namedtuple("Edge", ("connections", "exploit"))
    PreviousAction = namedtuple(
        "PreviousAction", ("host_name", "action_name", "success")
    )
    NodeFeatures = namedtuple(
        "Node", ("subnet", "relevance", "exploit", "malware", "prev_actuated")
    )

    # for encoding previous action ( imitates the logic in BlueTableWrapper._process_last_action() )
    global_actions_names = ("Sleep", "Monitor")
    active_actions = {"Restore": -1, "Remove": 1}  # any other = 0

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario=None,
        max_steps=100,
        render_mode=None,
        track_history=False,
    ) -> None:
        self.step_counter = None
        self.max_steps = max_steps
        self.track_history = track_history

        if not scenario:
            self.scenario_path = get_scenario(name="Scenario2", from_cyborg=True)
            self.scenario_name = "CybORG's Scenario2"
        else:
            self.scenario_path = get_scenario(name=scenario, from_cyborg=False)
            self.scenario_name = scenario

        self.cyborg = CybORG(self.scenario_path, "sim", agents={"Red": RedMeanderAgent})
        self.env_controller = self.cyborg.environment_controller
        self.scenario = self.env_controller.scenario

        self.blue_table = BlueTable(env=self.cyborg)
        self.red_table = None
        if self.track_history:
            self.red_table = RedTable(env=self.cyborg)

        self.host_names = self.scenario.hosts
        self.subnet_names = self.scenario.subnets
        self.action_names = self.scenario._scenario["Agents"][self.agent_name][
            "actions"
        ]

        self.hosts_exploited = set()
        self.hosts_with_malware = set()

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

        # Initialize feasiable connection graph with the structure from the scenario
        self.set_feasible_connections()

        # Set the relative importance of each host (by penalty of impact)
        self.set_host_relevance()

        # NOTE  the "true" state is not really updated because the observations are updated instead
        #       directly in self.env_controller.observation["Blue"]
        #       in the self.env_controller.step(...) method
        # state = ec.state
        # ec_st_true = ec.get_true_state(ec.INFO_DICT["True"]).data
        # ec_obs_true = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["True"])).data
        # ec_obs_blue = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["Blue"]), "Blue").data

        # ChallengeWrapper > OpenAIGymWrapper > EnumActionWrapper > BlueTableWrapper > TrueTableWrapper > CyBORG
        # self.challenge = ChallengeWrapper(
        #     agent_name=self.agent_name, env=self.cyborg, max_steps=self.max_steps
        # )
        # self.openai_gym = self.challenge.env
        # self.enum_action = self.openai_gym.env
        # self.blue_table = self.enum_action.env
        # self.true_table = self.blue_table.env

        # Set gymnasium properties
        # self.reward_range = (float("-inf"), float("inf"))  # not used
        self.action_space = gym.spaces.MultiDiscrete([self.num_hosts, self.num_actions])

        # NOTE  not very useful since unexpected connections appear regardless of layout constraints... and
        #       nested dict observations are not supported by stable baselines 3, only plain dicts
        # self.observation_space = self._build_dict_obs_space()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            # plt.ion()
            # plt.ioff()
            self.fig, self.axis = plt.subplots(1, 2)
            self._node_positions = plot_feasible_connections(self)

    def set_host_relevance(self):
        self.host_relevance = {}
        for hostname, host in self.scenario._scenario["Hosts"].items():
            confidentiality = host.get("ConfidentialityValue", None)
            availability = host.get("AvailabilityValue", None)

            if confidentiality == "None" and availability == "None":  # User0
                assert hostname == "User0"
                relevance = 0.0  # 0
            elif confidentiality is None and availability == "None":  # UsersX
                assert hostname.startswith("User")
                relevance = 0.1  # 1
            elif (
                confidentiality == "Medium" and availability == "Medium"
            ):  # EnterpriseX
                assert hostname.startswith("Enterprise")
                relevance = 1.0  # 2
            elif (
                confidentiality is None and availability is None
            ):  # Defender / Op_HostX
                assert hostname == "Defender" or hostname.startswith("Op_Host")
                relevance = 1.0  # 2
            elif confidentiality == "Medium" and availability == "High":  # Op_Server0
                assert hostname.startswith("Op_Server")
                relevance = 10.0  # 3
            else:
                raise ValueError(f"Unknown relevance for host {hostname}: {host}")

            self.host_relevance[hostname] = relevance

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
                    continue  # what is the meaning of this?

                origin_remote = (hostname, name)
                self.internet_connections.append(origin_remote)

                # set bidirectional connections
                remote_origin = (name, hostname)
                self.internet_connections.append(remote_origin)

        # hostname --> ip network (subnet)
        self.hostname_subnet_map = {}
        # hostname -->  subnet name
        self.hostname_subnetname_map = {}
        # subnet name --> hostnames
        self.subnet_hostnames_map = {}

        # NOTE: assuming bidirectional connections are possible
        #       between every pair of hosts in the same subnet
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

        global_signatures = [(None, action) for action in self.global_actions_names]

        host_actions = (
            action
            for action in self.action_names
            if action not in self.global_actions_names
        )
        action_host_signatures = product(self.host_names, host_actions)
        self.feasible_actions = list(action_host_signatures) + global_signatures

        # Equivalent to the logic in EnumActionWrapper.action_space_change(action_space_dict)
        # self.feasible_action_instances = list(starmap(self.instantiate_action, self.feasible_actions))

    def get_cyborg_action(self, host_idx, action_idx):
        "Converts gymnasium action to the equivalent cyborg action."
        action_name = self.action_names[action_idx]
        host_name = self.host_names[host_idx]
        if action_name in self.global_actions_names:
            LOGGER.debug(
                f"Global action {action_name} selected. Ignoring host selection {host_name}."
            )
            host_name = None  # global actions ignore host selection
        assert (host_name, action_name) in self.feasible_actions
        action_instance = instantiate_action(host_name, action_name)
        return action_instance

    def action_to_idx(self, cyborg_action):
        "Converts action name to the equivalent gymnasium action index."
        if cyborg_action is None:
            LOGGER.debug("None action selected. Using Sleep action.")
            return self.action_name_to_idx(None, "Sleep")
        host_name, action_name = get_action_names(cyborg_action)
        return self.action_name_to_idx(host_name, action_name)

    def action_name_to_idx(self, host_name, action_name):
        "Converts action name to the equivalent gymnasium action index."
        assert (host_name, action_name) in self.feasible_actions
        if host_name:
            host_idx = self.host_enumeration[host_name]
        else:
            host_idx = 0  # global actions ignore host selection
        action_idx = self.action_enumeration[action_name]
        return (host_idx, action_idx)

    def distill_observation(self, observation):
        """Extracts from the raw blue observation the information required
        to reconstruct the the blue table state but in a graph representation.
        """

        hosts_obs = {}
        connection_counter = defaultdict(int)
        exploit_connections = set()
        success = -1  # unknown or not set
        for host, properties in observation.items():
            if host == "success":
                success_enum = properties
                assert isinstance(success_enum, TrinaryEnum)

                if success_enum == TrinaryEnum.TRUE:
                    success = 1
                elif success_enum == TrinaryEnum.FALSE:
                    success = 0

                # NOTE The observation is valuable even if the previous action succeeded/failed
                continue

            num_local_ports = 0
            exploit = False
            if "Processes" in properties:
                processes = properties["Processes"]

                local_ports_counter = defaultdict(int)
                remote_ports_counter = defaultdict(int)

                for process in processes:
                    if "Connections" in process:
                        assert len(process["Connections"]) == 1
                        connection = process["Connections"][0]

                        if "Transport Protocol" in connection:
                            continue  # ignored by BlueTableWrapper

                        local_address = connection["local_address"]
                        remote_address = connection["remote_address"]
                        if local_address == remote_address:
                            # NOTE should self connections be included in the graph encoding?
                            LOGGER.debug(
                                f"Self-connection observed in {host}: {connection}"
                            )
                            # continue

                        local_host_name = self.hostname_ip_map.inv[local_address]
                        remote_host_name = self.hostname_ip_map.inv[remote_address]
                        local_remote_tuple = (local_host_name, remote_host_name)

                        assert host == local_host_name, "Utter nonsense again!"
                        # assert local_remote in self.feasible_connections, "Unfeasible connection appeared!"
                        if local_remote_tuple not in self.feasible_connections_set:
                            LOGGER.debug(
                                f"Unfeasible connection appeared! {local_host_name} --> {remote_host_name}"
                            )
                        connection_counter[local_remote_tuple] += 1

                        local_port = connection["local_port"]
                        local_ports_counter[local_port] += 1

                        try:
                            remote_port = connection["remote_port"]
                            remote_ports_counter[remote_port] += 1
                        except KeyError:
                            LOGGER.debug(
                                f"Connection {local_remote_tuple} has no remote port!"
                            )

                # BlueTable uses unique ports, so we use the number of unique ports

                # this is the total number of ports
                # num_local_ports = sum(local_ports_counter.values())

                # this is the number of unique ports
                num_local_ports = len(local_ports_counter)

                if (
                    4444 in remote_ports_counter
                    or len(remote_ports_counter) != num_local_ports
                ):
                    exploit = True
                    exploit_connections.add(local_remote_tuple)

            malware = False
            if "Files" in properties:
                files = properties["Files"]
                malware = any(_file["Density"] >= 0.9 for _file in files)

            hosts_obs[host] = self.HostObs(num_local_ports, exploit, malware)

        connections_obs = {}
        for connection, count in connection_counter.items():
            exploit = False
            if connection in exploit_connections:
                exploit = True
            connections_obs[connection] = self.EdgeObs(
                connections=count, exploit=exploit
            )

        # extract processes per host
        anomalies = self.blue_table._detect_anomalies(observation)
        # flag if processes represent a connection or a file
        relevant_anomalies = {
            host: processes
            for host, processes in anomalies.items()
            if "Connections" in processes.keys() or "Files" in processes.keys()
        }
        if relevant_anomalies:
            LOGGER.debug("Relevant anomalies detected:")
            LOGGER.debug(pformat(relevant_anomalies))

        previous_action = self.get_last_action()
        previous_host_name, previous_action_name = get_action_names(previous_action)
        previous_action = self.PreviousAction(
            previous_host_name, previous_action_name, success
        )

        return hosts_obs, connections_obs, previous_action

    def update_host_state(self, hosts_obs, previous_action):
        # first update the state based on the action taken if it was successful
        host_name = previous_action.host_name
        action_name = previous_action.action_name
        if action_name in self.active_actions and previous_action.success == 1:
            exploited_host = host_name in self.hosts_exploited
            privileged_host = host_name in self.hosts_with_malware
            if action_name == "Restore":  # always works
                if exploited_host:
                    self.hosts_exploited.remove(host_name)
                if privileged_host:
                    self.hosts_with_malware.remove(host_name)
            elif (
                action_name == "Remove"
            ):  # does not work if red agent has privileged access
                if exploited_host:
                    self.hosts_exploited.remove(host_name)

        # then update the state based on the observation
        for host_name, host_obs in hosts_obs.items():
            if host_obs.exploit:
                self.hosts_exploited.add(host_name)
            if host_obs.malware:
                self.hosts_with_malware.add(host_name)

    def encode_graph_observation(self, connection_obs, previous_action):
        """Transform the human understandable graph representation to a matrix encoding.
        Categorical values are not one-hot-encoded for now.
        """

        num_features = len(self.NodeFeatures._fields)
        node_matrix = np.zeros((self.num_hosts, num_features), dtype="i")  # int32
        for host_name in self.host_names:
            host_idx = self.host_enumeration[host_name]

            subnet_ip = self.hostname_subnet_map[host_name]
            subnet = self.subnet_cidr_map.inv[subnet_ip]
            subnet_id = self.subnet_enumeration[subnet]

            relevance = self.host_relevance[host_name]

            exploit = 0
            if host_name in self.hosts_exploited:
                exploit = 1

            malware = int(host_name in self.hosts_with_malware)

            prev_actuated = 0
            if host_name == previous_action.host_name:
                prev_actuated = self.active_actions.get(previous_action.action_name, 0)

            node_matrix[host_idx, :] = (
                subnet_id,
                relevance,
                exploit,
                malware,
                prev_actuated,
            )

        # This set difference needs to happen before any further access to the
        # connections object because it is a default dict and its keys change upon access
        unexpected_connections = connection_obs.keys() - self.feasible_connections_set

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

            edge_weight = connection_obs.get(
                (source, target), self.EdgeObs(connections=0, exploit=False)
            )
            edge_weights.append(
                edge_weight.connections
            )  # FIXME dropped flag for exploit in edges

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
                edge_weight = connection_obs.get(
                    (source, target), self.EdgeObs(connections=0, exploit=False)
                )
                extra_edge_weights.append(
                    edge_weight.connections
                )  # FIXME dropped flag for exploit in edges

            edge_tuples.extend(extra_edge_tuples)
            edge_weights.extend(extra_edge_weights)
            edge_index = np.hstack((edge_index, unexpected_edge_index))

        # edge weights are expected as a matrix of shape num_edges x num_attrs_per_edge
        edge_attr = np.array(edge_weights).reshape((-1, 1))

        success_encoding = torch.tensor([previous_action.success], dtype=torch.float)

        return Data(
            x=tensor(node_matrix, dtype=torch.float),
            edge_index=tensor(edge_index, dtype=torch.long),
            edge_attr=tensor(edge_attr, dtype=torch.float),
            global_attr=success_encoding,
        )

    def reset(self, *, seed=None):
        self.step_counter = 0

        # initialize previous action
        self.previous_action = self.PreviousAction(
            host_name=None,
            action_name="Sleep",
            success=-1,  # TrinaryEnum.UNKNOWN --> -1
        )

        # Subnet key error occurs when the Defender host is not initialized properly for some reason.
        # Resetting cyborg normally solves this issue on the first try.
        max_retries = 3
        for attempt in range(max_retries):
            try:
                cyborg_result = self.cyborg.reset(agent=self.agent_name, seed=seed)
                # patched BlueTable does not reset cyborg
                blue_table_obs = self.blue_table.reset(cyborg_result)
                break
            except KeyError as e:
                if e.args and e.args[0] == "Subnet":
                    LOGGER.error(
                        f"Subnet key error during reset (Attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt == max_retries:
                        LOGGER.exception(
                            f"Subnet key error persisted after {max_retries} attempts."
                        )
                        raise e

        # NOTE this depends on the random IPs assigned so need to be called after each environment reset
        self.set_feasible_connections()

        # extract graph represention of blue the initial observation of the blue agent
        host_obs, connections_obs, previous_action = self.distill_observation(
            self.get_raw_observation("Blue")
        )
        # self.update_host_state(host_obs, previous_action)  # not needed for initial observation
        observation = self.encode_graph_observation(
            connections_obs,
            previous_action,
        )

        info = {}
        if self.track_history:
            # patched RedTable does not reset cyborg
            red_table_obs = self.red_table.reset(cyborg_result)

            info["cyborg_result"] = vars(cyborg_result)

            info["true_state"] = self.get_true_state()
            info["true_table"] = self.get_true_table()

            info["blue_table"] = blue_table_obs
            info["blue_obs"] = self.get_raw_observation("Blue")

            info["red_table"] = red_table_obs
            info["red_obs"] = self.get_raw_observation("Red")

            graph_info = {
                "prev_action": previous_action,
                "hosts_obs": host_obs,
                "connections_obs": connections_obs,
                "exploited_hosts": self.hosts_exploited,
                "malware_hosts": self.hosts_with_malware,
            }
            info.update(graph_info)

        return observation, info

    def step(self, action):
        self.step_counter += 1

        action_instance = self.get_cyborg_action(*action)
        cyborg_result = self.cyborg.step(agent=self.agent_name, action=action_instance)

        # update host state based on the previous action

        host_obs, connections_obs, previous_action = self.distill_observation(
            self.get_raw_observation("Blue")  # == cyborg_result.observation
        )
        self.update_host_state(host_obs, previous_action)
        observation = self.encode_graph_observation(connections_obs, previous_action)

        info = {}
        if self.track_history:
            # NOTE: cyborg.step() call requires an agent name, which means the red table state
            #       update is manual and synced with the main cyborg instance at every step
            red_obs = self.get_raw_observation("Red")
            red_table_obs = self.red_table.observation_change(red_obs)

            # info["blue_obs"] = cyborg_result.observation  # already stored in "observation"
            blue_table_obs = self.blue_table.observation_change(
                cyborg_result.observation, baseline=False
            )

            info["cyborg_result"] = vars(cyborg_result)

            info["true_table"] = self.get_true_table()
            info["true_state"] = self.get_true_state()

            info["blue_table"] = blue_table_obs
            info["blue_obs"] = self.get_raw_observation("Blue")

            info["red_table"] = red_table_obs
            info["red_obs"] = self.get_raw_observation("Red")

            graph_info = {
                "prev_action": previous_action,
                "hosts_obs": host_obs,
                "connections_obs": connections_obs,
                "exploited_hosts": self.hosts_exploited,
                "malware_hosts": self.hosts_with_malware,
            }
            info.update(graph_info)

        reward = cyborg_result.reward

        terminated = cyborg_result.done

        truncated = False
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        # TODO add success status to plot
        host_obs, connections_obs, success = self.get_graph_observation()
        observation = self.encode_graph_observation(host_obs, connections_obs, success)
        if self.render_mode == "human":
            plot_observation(
                host_obs,
                connections_obs,
                axis=self.axis[0],
                node_positions=self._node_positions,
                show=True,
            )
            plot_observation_encoded(
                self,
                observation,
                node_positions=self._node_positions,
                axis=self.axis[1],
                show=True,
            )
            previous_action = self.get_last_action()
            if previous_action is None:
                self.fig.suptitle("Initial blue observation")
            else:
                self.fig.suptitle(f"Blue observation after {str(previous_action)}")
            self.fig.set_tight_layout(True)

    def get_encoded_observation(self):
        host_obs, connections_obs, success = self.get_graph_observation()
        return self.encode_graph_observation(host_obs, connections_obs, success)

    def get_graph_observation(self):
        raw_observation = self.get_raw_observation()
        return self.distill_observation(raw_observation)

    def get_raw_observation(self, agent=None):
        if agent is None:
            agent = self.agent_name
        return self.cyborg.get_observation(agent=agent)

    def get_true_state(self):
        return self.cyborg.get_agent_state("True")

    def get_true_table(self):
        # NOTE: true table is managed by the blue agent
        # return self.true_table.get_table()  # does not work
        return self.blue_table.get_table(output_mode="true_table")

    def get_blue_table(self):
        return self.blue_table.get_table(output_mode="blue_table")

    def get_red_table(self):
        if self.red_table is None:
            LOGGER.warning("Red table is not being tracked by the environment.")
            return None
        return self.red_table.get_table(output_mode="red_table")

    def get_last_action(self, agent=None):
        if agent is None:
            agent = self.agent_name
        return self.cyborg.get_last_action(agent)

    # NOTE use previous action in the graph repr with an independent linear transformation
    # def encode_last_action(self):
    #     """BlueTable logic relies only on the last action being of the broad type
    #     (Restore, Remove or Other)
    #     """
    #     action = self.get_last_action()
