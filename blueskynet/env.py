import inspect
from collections import defaultdict
from itertools import combinations

import gymnasium as gym
from bidict import frozenbidict

# https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html#torch-nn-functional-one-hot
# from torch.nn.functional import one_hot

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent # , TestAgent
from CybORG.Agents.Wrappers import ChallengeWrapper # , BaseWrapper


class GraphWrapper(gym.Env):  # , BaseWrapper

    # TODO define render mode
    agent_name = "Blue"

    def __init__(self, scenario="Scenario2", max_steps=100) -> None:

        self.scenario = scenario
        self.max_steps = max_steps

        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f"/Shared/Scenarios/{self.scenario}.yaml"

        # ChallengeWrapper > OpenAIGymWrapper > EnumActionWrapper > BlueTableWrapper > TrueTableWrapper > CyBORG
        self.cyborg = CybORG(path, "sim", agents={'Red': RedMeanderAgent})

        # NOTE  the "true" state is not really updated because the observations are updated instead...
        #       directly in ec.observation["Blue"] in the ec.step(...) method

        # state = ec.state
        # ec_st_true = ec.get_true_state(ec.INFO_DICT["True"]).data
        # ec_obs_true = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["True"])).data
        # ec_obs_blue = ec._filter_obs(ec.get_true_state(ec.INFO_DICT["Blue"]), "Blue").data

        self.env = ChallengeWrapper(agent_name=self.agent_name, env=self.cyborg, max_steps=self.max_steps)

        self.blue_table = self.env.env.env.env
        self.blue_baseline = self.blue_table.baseline

        # initialize connection map with communication structure from initial condition
        self.set_feasible_connections()

        self.host_props_baseline, self.connections_map = self.distill_observation(self.blue_baseline)

    def set_feasible_connections(self):
        "Extract graph layout from State object in CybORG, which is populated from the Scenario config."

        ec = self.cyborg.environment_controller

        # subnet name <--> ip network (subnet)
        self.subnet_cidr_map = frozenbidict(ec.subnet_cidr_map)
        # hostname <--> ip address
        self.hostname_ip_map = frozenbidict(ec.hostname_ip_map)
        # hostname --> ip network (subnet)
        # self.hostname_subnet_map = {
        #     host: props["Interface"][0]["Subnet"]
        #     for host, props in self.blue_baseline.items()
        #     if host != "sucess"
        # }
        # ip_address --> subnet
        # ec.state.get_subnet_containing_ip_address(ip_address)

        # feasible connections between hosts in different subnet
        self.internet_connections = []
        # ec.state.hosts --> dictionary from host names to Host() objects
        for hostname, host in ec.state.hosts.items():
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
        subnets = ec.state.subnets.values()
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

    def get_observation(self):
        # NOTE this is equivalent to: ec == CybORG.environment_controller and
        # ec.get_last_observation("Blue").data --> ec.observation["Blue"]
        blue_observation = self.blue_table.get_observation("Blue")
        return self.distill_observation(blue_observation)

    def get_table(self):
        self.blue_table.get_table()
        return self.blue_table.get_table()

    def get_last_action(self):
        return self.cyborg.get_last_action(self.agent_name)

    # TODO how to use previous action in the graph repr?
    # def encode_last_action(self):
    #     """BlueTable logic relies only on the last action being of the broad type
    #     (Restore, Remove or Other)
    #     """
    #     action = self.get_last_action()

