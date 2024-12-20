import os
import csv
import time

from logger import clear_, print_
from network import initialize_network
import numpy as np
import payments as ps
import routing as rt
import event as ev
import htlc as htlc


class NetworkParams:
    def __init__(self, n_nodes=100, n_channels=500, capacity_per_channel=1000,
                 faulty_node_prob=0.01, nodes_filename="nodes_ln.csv",
                 channels_filename="channels_ln.csv", edges_filename="edges_ln.csv",
                 network_from_file=False):
        self.n_nodes = n_nodes
        self.n_channels = n_channels
        self.capacity_per_channel = capacity_per_channel
        self.faulty_node_prob = faulty_node_prob
        self.nodes_filename = nodes_filename
        self.channels_filename = channels_filename
        self.edges_filename = edges_filename
        self.network_from_file = network_from_file

# Parameters for the payments
class PaymentsParams:
    def __init__(self, average_amount=1000, inverse_payment_rate=1.0,
                 n_payments=1000, payments_from_file=False,
                 payments_filename="payments.csv", mpp=0):
        self.average_amount = average_amount
        self.inverse_payment_rate = inverse_payment_rate
        self.n_payments = n_payments
        self.payments_from_file = payments_from_file
        self.payments_filename = payments_filename
        self.mpp = mpp


class Simulation:
    def __init__(self, current_time=0, events=None):
        self.current_time = current_time
        self.events = events
        self.random_generator = np.random.default_rng()

def write_output(network, payments, output_dir_name):
    if not os.path.exists(output_dir_name):
        print_("cloth.py: Cannot find the output directory. The output will be stored in the current directory.")
        output_dir_name = "./"

    with open(os.path.join(output_dir_name, "channels_output.csv"), "w", newline='') as csv_channel_output:
        writer = csv.writer(csv_channel_output)
        writer.writerow(["id", "edge1", "edge2", "node1", "node2", "capacity", "is_closed"])
        for channel in network.channels:
            writer.writerow([channel.id, channel.edge1, channel.edge2, channel.node1, channel.node2, channel.capacity, channel.is_closed])

    with open(os.path.join(output_dir_name, "edges_output.csv"), "w", newline='') as csv_edge_output:
        writer = csv.writer(csv_edge_output)
        writer.writerow(["id", "channel_id", "counter_edge_id", "from_node_id", "to_node_id", "balance", "fee_base", "fee_proportional", "min_htlc", "timelock", "is_closed", "tot_flows"])
        for edge in network.edges:
            writer.writerow([edge.id, edge.channel_id, edge.counter_edge_id, edge.from_node_id, edge.to_node_id, edge.balance, edge.policy.fee_base, edge.policy.fee_proportional, edge.policy.min_htlc, edge.policy.timelock, edge.is_closed, edge.tot_flows])

    with open(os.path.join(output_dir_name, "payments_output.csv"), "w", newline='') as csv_payment_output:
        writer = csv.writer(csv_payment_output)
        writer.writerow(["id", "sender_id", "receiver_id", "amount", "start_time", "end_time", "mpp", "is_success", "no_balance_count", "offline_node_count", "timeout_exp", "attempts", "route", "total_fee"])
        for payment in payments:
            if payment.id == -1:
                continue
            row = [payment.id, payment.sender, payment.receiver, payment.amount, payment.start_time, payment.end_time, payment.is_shard, payment.is_success, payment.no_balance_count, payment.offline_node_count, payment.is_timeout, payment.attempts]
            if payment.route is None:
                row.append(-1)
            else:
                hops = payment.route.route_hops
                route_hops = '-'.join(str(hop.edge_id) for hop in hops)
                row.append(route_hops)
                row.append(payment.route.total_fee)
            writer.writerow(row)

    with open(os.path.join(output_dir_name, "nodes_output.csv"), "w", newline='') as csv_node_output:
        writer = csv.writer(csv_node_output)
        writer.writerow(["id", "open_edges"])
        for node in network.nodes:
            row = [node.id]
            if len(node.open_edges) == 0:
                row.append(-1)
            else:
                open_edges = '-'.join(str(id_) for id_ in node.open_edges)
                row.append(open_edges)
            writer.writerow(row)

def initialize_input_parameters(net_params, pay_params):
    net_params.n_nodes = net_params.n_channels = net_params.capacity_per_channel = 0
    net_params.faulty_node_prob = 0.0
    net_params.network_from_file = 0
    net_params.nodes_filename = ""
    net_params.channels_filename = ""
    net_params.edges_filename = ""
    pay_params.inverse_payment_rate = pay_params.average_amount = 0.0
    pay_params.n_payments = 0
    pay_params.payments_from_file = 0
    pay_params.payments_filename = ""
    pay_params.mpp = 0

def read_input(net_params, pay_params):
    initialize_input_parameters(net_params, pay_params)

    try:
        with open("cloth_input.txt", "r") as input_file:
            for line in input_file:
                key, value = line.strip().split("=", 1)
                if not value:  # If value is missing, skip or set default
                    continue

                if key == "generate_network_from_file":
                    net_params.network_from_file = value.lower() == "true"
                elif key == "nodes_filename":
                    net_params.nodes_filename = value
                elif key == "channels_filename":
                    net_params.channels_filename = value
                elif key == "edges_filename":
                    net_params.edges_filename = value
                elif key == "n_additional_nodes":
                    net_params.n_nodes = int(value) if value else net_params.n_nodes
                elif key == "n_channels_per_node":
                    net_params.n_channels = int(value) if value else net_params.n_channels
                elif key == "capacity_per_channel":
                    net_params.capacity_per_channel = int(value) if value else net_params.capacity_per_channel
                elif key == "faulty_node_probability":
                    net_params.faulty_node_prob = float(value) if value else net_params.faulty_node_prob
                elif key == "generate_payments_from_file":
                    pay_params.payments_from_file = value.lower() == "true"
                elif key == "payments_filename":
                    pay_params.payments_filename = value
                elif key == "payment_rate":
                    pay_params.payment_rate = float(value) if value else pay_params.payment_rate
                elif key == "n_payments":
                    pay_params.n_payments = int(value) if value else pay_params.n_payments
                elif key == "average_payment_amount":
                    pay_params.average_amount = int(value) if value else pay_params.average_amount
                elif key == "mpp":
                    pay_params.mpp = int(value) if value else pay_params.mpp
                else:
                    raise ValueError(f"Unknown parameter {key}")
    except FileNotFoundError:
        print_("ERROR: cannot open file <cloth_input.txt> in current directory.")
        exit(-1)

def has_shards(payment):
    return payment.shards_id[0] != -1 and payment.shards_id[1] != -1

def post_process_payment_stats(payments):
    for payment in payments:
        if payment.id == -1 or not has_shards(payment):
            continue
        shard1 = payments[payment.shards_id[0]]
        shard2 = payments[payment.shards_id[1]]
        payment.end_time = max(shard1.end_time, shard2.end_time)
        payment.is_success = int(shard1.is_success and shard2.is_success)
        payment.no_balance_count = shard1.no_balance_count + shard2.no_balance_count
        payment.offline_node_count = shard1.offline_node_count + shard2.offline_node_count
        payment.is_timeout = int(shard1.is_timeout or shard2.is_timeout)
        payment.attempts = shard1.attempts + shard2.attempts
        if shard1.route is not None and shard2.route is not None:
            payment.route = shard1.route if len(shard1.route.route_hops) > len(shard2.route.route_hops) else shard2.route
            payment.route.total_fee = shard1.route.total_fee + shard2.route.total_fee
        else:
            payment.route = None
        shard1.id = -1
        shard2.id = -1

def initialize_random_generator():
    return np.random.default_rng()

def main(argv):
    clear_()
    if len(argv) != 2:
        print_("ERROR cloth.py: please specify the output directory")
        return -1

    output_dir_name = argv[1]
    net_params = NetworkParams(network_from_file=True)
    pay_params = PaymentsParams(payments_from_file=True)
    read_input(net_params, pay_params)

    simulation = Simulation()
    simulation.random_generator = initialize_random_generator()
    print_("NETWORK INITIALIZATION")
    network = initialize_network(net_params)
    n_nodes = len(network.nodes)
    n_edges = len(network.edges)
    print_("PAYMENTS INITIALIZATION")
    payments = ps.initialize_payments(pay_params, n_nodes, simulation.random_generator)
    print_("EVENTS INITIALIZATION")
    simulation.events = ev.initialize_events(payments)
    rt.initialize_dijkstra(n_nodes, n_edges, payments)
    htlc.initialize_paths(payments, network)

    print_("INITIAL DIJKSTRA THREADS EXECUTION")
    start = time.time()
    rt.run_dijkstra_threads(network, payments, 0)
    time_spent_thread = time.time() - start
    print_(f"Time consumed by initial dijkstra executions: {time_spent_thread:.2f} s")

    print_("EXECUTION OF THE SIMULATION")
    begin = time.time()
    simulation.current_time = 1
    while simulation.events.length() != 0:
        event = simulation.events.pop(ev.compare_event)
        simulation.current_time = event.time
        if event.type == ev.EventType.FINDPATH:
            htlc.find_path(event, simulation, network, payments, pay_params.mpp == 1)
        elif event.type == ev.EventType.SENDPAYMENT:
            htlc.send_payment(event, simulation, network)
        elif event.type == ev.EventType.FORWARDPAYMENT:
            htlc.forward_payment(event, simulation, network)
        elif event.type == ev.EventType.RECEIVEPAYMENT:
            htlc.receive_payment(event, simulation, network)
        elif event.type == ev.EventType.FORWARDSUCCESS:
            htlc.forward_success(event, simulation, network)
        elif event.type == ev.EventType.RECEIVESUCCESS:
            htlc.receive_success(event, simulation, network)
        elif event.type == ev.EventType.FORWARDFAIL:
            htlc.forward_fail(event, simulation, network)
        elif event.type == ev.EventType.RECEIVEFAIL:
            htlc.receive_fail(event, simulation, network)
        elif event.type == ev.EventType.OPENCHANNEL:
            htlc.open_channel(network, simulation.random_generator)
        else:
            print_(event.type)
            print_("ERROR wrong event type")
            exit(-1)

    if pay_params.mpp:
        post_process_payment_stats(payments)

    time_spent = time.time() - begin
    print_(f"Time consumed by simulation events: {time_spent:.2f} s")

    write_output(network, payments, output_dir_name)

if __name__ == "__main__":
    import sys
    main(sys.argv)

