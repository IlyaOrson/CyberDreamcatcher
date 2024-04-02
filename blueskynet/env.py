import sys
from collections import defaultdict
from itertools import combinations, product  #, starmap

import numpy as np
import gymnasium as gym
from bidict import bidict

# https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html#torch-nn-functional-one-hot
# from torch.nn.functional import one_hot

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent # , TestAgent
from CybORG.Agents.Wrappers import ChallengeWrapper  #, BaseWrapper

from blueskynet.utils import get_scenario

# We do not inherit from wrapper because we need lower level information for our graph
# than the distilled information that travels through the wrappers observations
class GraphWrapper(gym.Env):

    # TODO define render mode
    agent_name = "Blue"

    def __init__(self, scenario=None, max_steps=100) -> None:

        self.max_steps = max_steps

        if not scenario:
            self.scenario_path = get_scenario(name="Scenario2", from_cyborg=True)
        else:
            self.scenario_path = get_scenario(name=scenario, from_cyborg=False)

        self.cyborg = CybORG(self.scenario_path, "sim", agents={"Red": RedMeanderAgent})
        self.env_controller = self.cyborg.environment_controller

        # NOTE  the "true" state is not really updated because the observations are updated instead
        #       directly in self.env_controller.observation["Blue"]
        #       in the self.env_controller.step(...) method
        # state = ec.state
        # ec_st_true = ec.get_true_state(ec.INFO_DICT["True"]).data
        # ec_obs_true = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["True"])).data
        # ec_obs_blue = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["Blue"]), "Blue").data

        # ChallengeWrapper > OpenAIGymWrapper > EnumActionWrapper > BlueTableWrapper > TrueTableWrapper > CyBORG
        self.challenge = ChallengeWrapper(agent_name=self.agent_name, env=self.cyborg, max_steps=self.max_steps)
        self.openai_gym = self.challenge.env
        self.enum_action = self.openai_gym.env
        self.blue_table = self.enum_action.env
        self.true_table = self.blue_table.env

        # This imitates the logic in BlueTable._process_initial_obs()
        self.blue_baseline = {k: v for k,v in self.get_observation().items() if k != "success"}

        # Initialize feasiable connection graph with the structure from the scenario
        self.set_feasible_connections()

        # Extract graph represention of blue the initial observation of the blue agent
        self.host_props_baseline, self.observed_connections = self.distill_observation(self.blue_baseline)

        # Extract possible actions from cyborg
        # action_space = self.env_controller.agent_interfaces[self.agent_name].action_space
        self.action_names = self.env_controller.scenario._scenario["Agents"][self.agent_name]["actions"]
        self.set_feasible_actions()

        self.action_space = gym.spaces.MultiDiscrete([len(self.action_names), len(self.host_names)])

    def set_feasible_connections(self):
        "Extract graph layout from State object in CybORG, which is populated from the Scenario config."

        # subnet name <--> ip network (subnet)
        self.subnet_cidr_map = bidict(self.env_controller.subnet_cidr_map)
        # hostname <--> ip address
        self.hostname_ip_map = bidict(self.env_controller.hostname_ip_map)
        # ip_address --> subnet
        # self.env_controller.state.get_subnet_containing_ip_address(ip_address)

        # feasible connections between hosts in different subnet
        self.internet_connections = []
        # self.env_controller.state.hosts --> dictionary from host names to Host() objects
        for hostname, host in self.env_controller.state.hosts.items():
            # self_reference = False
            # if interface appears here, suppose a connection is feasible
            for name, interface in host.info.items():
                if name == hostname:
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

        self.host_names = list(self.hostname_ip_map.keys())
        self.subnet_names = list(self.subnet_cidr_map.keys())
        self.feasible_connections = set(self.intranet_connections + self.internet_connections)

    def distill_observation(self, observation):
        """Extracts from the blue observation the information required
        to reconstruct the the blue table state but in a graph representation
        """

        host_properties = {}
        connections_between_hosts = defaultdict(int)
        for host, properties in observation.items():

            if host == "success":
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
                            continue  # FIXME double check

                        local_port = connection["local_ports"]
                        local_ports_counter[local_port] += 1

                        remote_port = connection["remote_ports"]
                        remote_ports_counter[remote_port] += 1

                        local_address = connection["local_address"]
                        remote_address = connection["remote_address"]
                        local_remote = (local_address, remote_address)
                        assert local_remote in self.feasible_connections
                        connections_between_hosts[local_remote] += 1

                        assert local_address == self.subnet_cidr_map[host]

                num_local_ports = sum(local_ports_counter.values())

            malware = False
            if "Files" in properties:
                files = properties["Files"]
                malware = any(_file['Density'] >= 0.9 for _file in files)

            subnet_ip = self.hostname_subnet_map[host]
            subnet = self.subnet_cidr_map.inv[subnet_ip]
            host_properties[host] = [subnet, num_local_ports, malware]

        return host_properties, connections_between_hosts

    # TODO method to encode observation

    def get_action_class(self, action_name):
        "Retrieve the action class defined in CybORG from the action name."
        action_module = sys.modules["CybORG.Shared.Actions"]
        action_class = getattr(action_module, action_name)
        return action_class

    def gym_to_cyborg_action(self, gym_action):
        "Converts gymnasium action to the equivalent cyborg action."
        action_idx, host_idx = gym_action
        action_name = self.action_names[action_idx]
        host_name = self.host_names[host_idx] 
        if action_name in self.global_actions_names:
            host_name = None  # global actions ignore host selection
        assert (action_name, host_name) in self.feasible_actions
        action_instance = self.instantiate_action(action_name, host_name)
        return action_instance

    def instantiate_action(self, action_name, host_name):
        """Create instantiate the class object with the given host."""

        action_class = self.get_action_class(action_name)

        if action_name == "Sleep":
            action = action_class()
        elif action_name == "Monitor":
            action = action_class(session=0, agent=self.agent_name)
        else:
            action = action_class(session=0, agent=self.agent_name, hostname=host_name)
        return action

    def set_feasible_actions(self):
        """Iterate over the action classes reported by cyborg and instantiate each of them
        with the parameters available in the action space which match their signature.
        """

        self.global_actions_names = ("Sleep", "Monitor")
        global_signatures = [(action, None) for action in self.global_actions_names]

        host_actions = (action for action in self.action_names if action not in self.global_actions_names)
        action_host_signatures = product(host_actions, self.host_names)
        self.feasible_actions = list(action_host_signatures) + global_signatures

        # Equivalent to the logic in EnumActionWrapper.action_space_change(action_space_dict)
        # self.feasible_action_instances = list(starmap(self.instantiate_action, self.feasible_actions))

    def reset(self, *, seed = None, options = None):
        # CybORG does not expect options as a keyword
        if options:
            result = self.cyborg.reset(seed=seed, **options)
        else:
            result = self.cyborg.reset(seed=seed)

        return np.array(result.observation), vars(result)

    def step(self, action):
        action_instance = self.gym_to_cyborg_action(action)
        cyborg_result = self.cyborg.step(agent=self.agent_name, action=action_instance)
        obs = np.array(cyborg_result.observation)  # TODO gymnasium space
        reward = cyborg_result.reward
        terminated = cyborg_result.done
        truncated = False
        info = vars(cyborg_result)
        return obs, reward, terminated, truncated, info

    def get_observation(self):
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
