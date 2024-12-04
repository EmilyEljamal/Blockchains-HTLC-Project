import numpy as np
import pandas as pd
import random
from array_ import Array
from list_ import Element
from typing import List, Optional

# Constants
MAXMSATOSHI = 5e17
MAXTIMELOCK = 100
MINTIMELOCK = 10
MAXFEEBASE = 5000
MINFEEBASE = 1000
MAXFEEPROP = 10
MINFEEPROP = 1
MAXLATENCY = 100
MINLATENCY = 10
MINBALANCE = 1e2
MAXBALANCE = 1e11

class Policy:
    def __init__(self, fee_base, fee_proportional, min_htlc, timelock):
        self.fee_base = fee_base
        self.fee_proportional = fee_proportional
        self.min_htlc = min_htlc
        self.timelock = timelock

class Node:
    def __init__(self, id_):
        self.id = id_
        self.open_edges = Array(10)
        self.results: List[Optional[Element]] = [None]
        self.explored = False

class Channel:
    def __init__(self, id_, direction1, direction2, node1, node2, capacity):
        self.id = id_
        self.edge1 = direction1
        self.edge2 = direction2
        self.node1 = node1
        self.node2 = node2
        self.capacity = capacity
        self.is_closed = False

class Edge:
    def __init__(self, id_, channel_id, counter_edge_id, from_node_id, to_node_id, balance, policy):
        self.id = id_
        self.channel_id = channel_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.counter_edge_id = counter_edge_id
        self.policy = policy
        self.balance = balance
        self.is_closed = False
        self.tot_flows = 0

class Network:
    def __init__(self):
        self.nodes = Array(1000)
        self.channels = Array(1000)
        self.edges = Array(2000)
        self.faulty_node_prob = None
        self.final_timelock = MAXTIMELOCK

# Write Network Files
def write_network_files(network: Network):
    with open("nodes.csv", 'w') as nodes_file, \
         open("channels.csv", 'w') as channels_file, \
         open("edges.csv", 'w') as edges_file:

        # Nodes
        nodes_file.write("id\n")
        for i in range(network.nodes.length()):
            node = network.nodes.get(i)
            nodes_file.write(f"{node.id}\n")

        # Channels
        channels_file.write("id,edge1_id,edge2_id,node1_id,node2_id,capacity\n")
        for i in range(network.channels.length()):
            channel = network.channels.get(i)
            channels_file.write(f"{channel.id},{channel.edge1},{channel.edge2},{channel.node1},{channel.node2},{channel.capacity}\n")

        # Edges
        edges_file.write("id,channel_id,counter_edge_id,from_node_id,to_node_id,balance,fee_base,fee_proportional,min_htlc,timelock\n")
        for i in range(network.edges.length()):
            edge = network.edges.get(i)
            policy = edge.policy
            edges_file.write(f"{edge.id},{edge.channel_id},{edge.counter_edge_id},{edge.from_node_id},{edge.to_node_id},{edge.balance},{policy.fee_base},{policy.fee_proportional},{policy.min_htlc},{policy.timelock}\n")

# Update Probability Per Node
def update_probability_per_node(probability_per_node, channels_per_node, n_nodes, node1_id, node2_id, tot_channels):
    channels_per_node[node1_id] += 1
    channels_per_node[node2_id] += 1
    for i in range(n_nodes):
        probability_per_node[i] = channels_per_node[i] / tot_channels

# Generate Random Channel
def generate_random_channel(channel_data: dict, mean_channel_capacity, network: Network):
    capacity = abs(mean_channel_capacity + np.random.normal())
    channel = Channel(channel_data["id"], channel_data["edge1"], channel_data["edge2"], channel_data["node1"], channel_data["node2"], capacity * 1000)

    fraction_capacity = random.random()
    edge1_balance = fraction_capacity * capacity
    edge2_balance = capacity - edge1_balance
    edge1_balance *= 1000
    edge2_balance *= 1000

    edge1_policy = Policy(
        fee_base=random.randint(MINFEEBASE, MAXFEEBASE),
        fee_proportional=random.randint(MINFEEPROP, MAXFEEPROP),
        timelock=random.randint(MINTIMELOCK, MAXTIMELOCK),
        min_htlc=10 ** random.choice([0, 1, 2, 3])
    )
    edge2_policy = Policy(
        fee_base=random.randint(MINFEEBASE, MAXFEEBASE),
        fee_proportional=random.randint(MINFEEPROP, MAXFEEPROP),
        timelock=random.randint(MINTIMELOCK, MAXTIMELOCK),
        min_htlc=10 ** random.choice([0, 1, 2, 3])
    )

    edge1 = Edge(channel_data["edge1"], channel_data["id"], channel_data["edge2"], channel_data["node1"], channel_data["node2"], edge1_balance, edge1_policy)
    edge2 = Edge(channel_data["edge2"], channel_data["id"], channel_data["edge1"], channel_data["node2"], channel_data["node1"], edge2_balance, edge2_policy)

    network.channels.insert(channel)
    network.edges.insert(edge1)
    network.edges.insert(edge2)

    node1 = network.nodes.get(channel_data["node1"])
    node1.open_edges.insert(edge1.id)
    node2 = network.nodes.get(channel_data["node2"])
    node2.open_edges.insert(edge2.id)

# Generate Random Network
def generate_random_network(net_params):
    network = Network()
    rng = np.random.default_rng()
    for i in range(net_params['n_channels']):
        channel_data = {
            'id': i,
            'edge1': len(network.edges),
            'edge2': len(network.edges) + 1,
            'node1': rng.integers(0, len(network.nodes)),
            'node2': rng.integers(0, len(network.nodes))
        }
        generate_random_channel(channel_data, net_params['capacity_per_channel'], network)

    write_network_files(network)
    return network

# Initialize Network
def initialize_network(net_params):
    if net_params.network_from_file:
        network = generate_network_from_files(
            net_params.nodes_filename,
            net_params.channels_filename,
            net_params.edges_filename
        )
    else:
        network = generate_random_network({
            'n_nodes': net_params.n_nodes,
            'n_channels': net_params.n_channels,
            'capacity_per_channel': net_params.capacity_per_channel,
            'faulty_node_prob': net_params.faulty_node_prob
        })

    n_nodes = len(network.nodes)
    for node in network.nodes:
        node.results = [None] * n_nodes
    return network

def generate_network_from_files(nodes_filename: str, channels_filename: str, edges_filename: str) -> Network:
    network = Network()

    # Read nodes
    nodes_df = pd.read_csv(nodes_filename)
    for _, row in nodes_df.iterrows():
        node = Node(id_=row['id'])
        network.nodes.insert(node)

    # Read channels
    channels_df = pd.read_csv(channels_filename)
    for _, row in channels_df.iterrows():
        channel = Channel(
            id_=row['id'],
            direction1=row['edge1_id'],
            direction2=row['edge2_id'],
            node1=row['node1_id'],
            node2=row['node2_id'],
            capacity=row['capacity']
        )
        network.channels.insert(channel)

    # Read edges
    edges_df = pd.read_csv(edges_filename)
    for _, row in edges_df.iterrows():
        policy = Policy(
            fee_base=row['fee_base'],
            fee_proportional=row['fee_proportional'],
            min_htlc=row['min_htlc'],
            timelock=row['timelock']
        )
        edge = Edge(
            id_=row['id'],
            channel_id=row['channel_id'],
            counter_edge_id=row['counter_edge_id'],
            from_node_id=row['from_node_id'],
            to_node_id=row['to_node_id'],
            balance=row['balance'],
            policy=policy
        )
        network.edges.insert(edge)

        from_node = network.nodes.get(edge.from_node_id)
        to_node = network.nodes.get(edge.to_node_id)
        if from_node:
            from_node.open_edges.insert(edge.id)
        if to_node:
            to_node.open_edges.insert(edge.id)

    return network
