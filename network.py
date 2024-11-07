import numpy as np
import pandas as pd
import random

class Node:
    def __init__(self, id):
        self.id = id
        self.open_edges = []
        self.results = None
        self.explored = 0

class Channel:
    def __init__(self, id, direction1, direction2, node1, node2, capacity):
        self.id = id
        self.edge1 = direction1
        self.edge2 = direction2
        self.node1 = node1
        self.node2 = node2
        self.capacity = capacity
        self.is_closed = 0

class Edge:
    def __init__(self, id, channel_id, counter_edge_id, from_node_id, to_node_id, balance, policy):
        self.id = id
        self.channel_id = channel_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.counter_edge_id = counter_edge_id
        self.policy = policy
        self.balance = balance
        self.is_closed = 0
        self.tot_flows = 0

class Network:
    def __init__(self):
        self.nodes = []
        self.channels = []
        self.edges = []
        self.faulty_node_prob = None

def write_network_files(network):
    nodes_output_file = "nodes.csv"
    channels_output_file = "channels.csv"
    edges_output_file = "edges.csv"

    with open(nodes_output_file, 'w') as f:
        f.write("id\n")
        for node in network.nodes:
            f.write(f"{node.id}\n")

    with open(channels_output_file, 'w') as f:
        f.write("id,edge1_id,edge2_id,node1_id,node2_id,capacity\n")
        for channel in network.channels:
            f.write(f"{channel.id},{channel.edge1},{channel.edge2},{channel.node1},{channel.node2},{channel.capacity}\n")

    with open(edges_output_file, 'w') as f:
        f.write("id,channel_id,counter_edge_id,from_node_id,to_node_id,balance,fee_base,fee_proportional,min_htlc,timelock\n")
        for edge in network.edges:
            f.write(f"{edge.id},{edge.channel_id},{edge.counter_edge_id},{edge.from_node_id},{edge.to_node_id},{edge.balance},{edge.policy['fee_base']},{edge.policy['fee_proportional']},{edge.policy['min_htlc']},{edge.policy['timelock']}\n")

def update_probability_per_node(probability_per_node, channels_per_node, n_nodes, node1_id, node2_id, tot_channels):
    channels_per_node[node1_id] += 1
    channels_per_node[node2_id] += 1
    for i in range(n_nodes):
        probability_per_node[i] = channels_per_node[i] / tot_channels

def generate_random_channel(channel_data, mean_channel_capacity, network):
    capacity = abs(mean_channel_capacity + np.random.normal())
    channel = Channel(channel_data.id, channel_data.edge1, channel_data.edge2, channel_data.node1, channel_data.node2, capacity * 1000)

    fraction_capacity = random.random()
    edge1_balance = fraction_capacity * capacity
    edge2_balance = capacity - edge1_balance
    edge1_balance *= 1000
    edge2_balance *= 1000

    edge1_policy = {
        'fee_base': random.randint(MINFEEBASE, MAXFEEBASE),
        'fee_proportional': random.randint(MINFEEPROP, MAXFEEPROP),
        'timelock': random.randint(MINTIMELOCK, MAXTIMELOCK),
        'min_htlc': 10 ** random.choice([0, 1, 2, 3])  # Simplified for example
    }
    edge1_policy['min_htlc'] = 0 if edge1_policy['min_htlc'] == 1 else edge1_policy['min_htlc']

    edge2_policy = {
        'fee_base': random.randint(MINFEEBASE, MAXFEEBASE),
        'fee_proportional': random.randint(MINFEEPROP, MAXFEEPROP),
        'timelock': random.randint(MINTIMELOCK, MAXTIMELOCK),
        'min_htlc': 10 ** random.choice([0, 1, 2, 3])  # Simplified for example
    }
    edge2_policy['min_htlc'] = 0 if edge2_policy['min_htlc'] == 1 else edge2_policy['min_htlc']

    edge1 = Edge(channel_data.edge1, channel_data.id, channel_data.edge2, channel_data.node1, channel_data.node2, edge1_balance, edge1_policy)
    edge2 = Edge(channel_data.edge2, channel_data.id, channel_data.edge1, channel_data.node2, channel_data.node1, edge2_balance, edge2_policy)

    network.channels.append(channel)
    network.edges.append(edge1)
    network.edges.append(edge2)

    node1 = network.nodes[channel_data.node1]
    node1.open_edges.append(edge1.id)
    node2 = network.nodes[channel_data.node2]
    node2.open_edges.append(edge2.id)

def generate_random_network(net_params):
    network = Network()
    network.nodes = [Node(i) for i in range(1000)]
    network.channels = []
    network.edges = []

    # Simulate reading from files and generating channels
    for i in range(net_params['n_channels']):
        channel_data = {
            'id': i,
            'edge1': len(network.edges),
            'edge2': len(network.edges) + 1,
            'node1': random.randint(0, len(network.nodes) - 1),
            'node2': random.randint(0, len(network.nodes) - 1)
        }
        generate_random_channel(channel_data, net_params['capacity_per_channel'], network)

    write_network_files(network)
    return network

def generate_network_from_files(nodes_filename, channels_filename, edges_filename):
    network = Network()
    network.nodes = []
    network.channels = []
    network.edges = []

    nodes_df = pd.read_csv(nodes_filename)
    for _, row in nodes_df.iterrows():
        node = Node(row['id'])
        network.nodes.append(node)

    channels_df = pd.read_csv(channels_filename)
    for _, row in channels_df.iterrows():
        channel = Channel(row['id'], row['edge1_id'], row['edge2_id'], row['node1_id'], row['node2_id'], row['capacity'])
        network.channels.append(channel)

    edges_df = pd.read_csv(edges_filename)
    for _, row in edges_df.iterrows():
        policy = {
            'fee_base': row['fee_base'],
            'fee_proportional': row['fee_proportional'],
            'min_htlc': row['min_htlc'],
            'timelock': row['timelock']
        }
        edge = Edge(row['id'], row['channel_id'], row['counter_edge_id'], row['from_node_id'], row['to_node_id'], row['balance'], policy)
        network.edges.append(edge)

    return network

def initialize_network(net_params):
    if net_params['network_from_file']:
        network = generate_network_from_files(net_params['nodes_filename'], net_params['channels_filename'], net_params['edges_filename'])
    else:
        network = generate_random_network(net_params)

    n_nodes = len(network.nodes)
    for node in network.nodes:
        node.results = [None] * n_nodes

    return network

def open_channel(network):
    channel = Channel(len(network.channels), len(network.edges), len(network.edges) + 1, random.randint(0, len(network.nodes) - 1), random.randint(0, len(network.nodes) - 1), 1000)
    generate_random_channel(channel, 1000, network)

