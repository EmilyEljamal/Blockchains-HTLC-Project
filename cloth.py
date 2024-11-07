import os
import csv
import time
import numpy as np

def write_output(network, payments, output_dir_name):
    if not os.path.exists(output_dir_name):
        print("cloth.py: Cannot find the output directory. The output will be stored in the current directory.")
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
                open_edges = '-'.join(str(id) for id in node.open_edges)
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
                parameter, value = line.strip().split("=")
                parameter = parameter.strip()
                value = value.strip()

                if parameter == "generate_network_from_file":
                    net_params.network_from_file = 1 if value == "true" else 0
                elif parameter == "nodes_filename":
                    net_params.nodes_filename = value
                elif parameter == "channels_filename":
                    net_params.channels_filename = value
                elif parameter == "edges_filename":
                    net_params.edges_filename = value
                elif parameter == "n_additional_nodes":
                    net_params.n_nodes = int(value)
                elif parameter == "n_channels_per_node":
                    net_params.n_channels = int(value)
                elif parameter == "capacity_per_channel":
                    net_params.capacity_per_channel = int(value)
                elif parameter == "faulty_node_probability":
                    net_params.faulty_node_prob = float(value)
                elif parameter == "generate_payments_from_file":
                    pay_params.payments_from_file = 1 if value == "true" else 0
                elif parameter == "payments_filename":
                    pay_params.payments_filename = value
                elif parameter == "payment_rate":
                    pay_params.inverse_payment_rate = 1.0 / float(value)
                elif parameter == "n_payments":
                    pay_params.n_payments = int(value)
                elif parameter == "average_payment_amount":
                    pay_params.average_amount = float(value)
                elif parameter == "mpp":
                    pay_params.mpp = int(value)
                else:
                    raise ValueError(f"Unknown parameter {parameter}")
    except FileNotFoundError:
        print("ERROR: cannot open file <cloth_input.txt> in current directory.")
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
    if len(argv) != 2:
        print("ERROR cloth.py: please specify the output directory")
        return -1

    output_dir_name = argv[1]
    net_params = NetworkParams()
    pay_params = PaymentsParams()
    read_input(net_params, pay_params)

    simulation = Simulation()
    simulation.random_generator = initialize_random_generator()
    print("NETWORK INITIALIZATION")
    network = initialize_network(net_params, simulation.random_generator)
    n_nodes = len(network.nodes)
    n_edges = len(network.edges)
    print("PAYMENTS INITIALIZATION")
    payments = initialize_payments(pay_params, n_nodes, simulation.random_generator)
    print("EVENTS INITIALIZATION")
    simulation.events = initialize_events(payments)
    initialize_dijkstra(n_nodes, n_edges, payments)

    print("INITIAL DIJKSTRA THREADS EXECUTION")
    start = time.time()
    run_dijkstra_threads(network, payments, 0)
    time_spent_thread = time.time() - start
    print(f"Time consumed by initial dijkstra executions: {time_spent_thread:.2f} s")

    print("EXECUTION OF THE SIMULATION")
    begin = time.time()
    simulation.current_time = 1
    while len(simulation.events) != 0:
        event = heap_pop(simulation.events, compare_event)
        simulation.current_time = event.time
        if event.type == FINDPATH:
            find_path(event, simulation, network, payments, pay_params.mpp)
        elif event.type == SENDPAYMENT:
            send_payment(event, simulation, network)
        elif event.type == FORWARDPAYMENT:
            forward_payment(event, simulation, network)
        elif event.type == RECEIVEPAYMENT:
            receive_payment(event, simulation, network)
        elif event.type == FORWARDSUCCESS:
            forward_success(event, simulation, network)
        elif event.type == RECEIVESUCCESS:
            receive_success(event, simulation, network)
        elif event.type == FORWARDFAIL:
            forward_fail(event, simulation, network)
        elif event.type == RECEIVEFAIL:
            receive_fail(event, simulation, network)
        elif event.type == OPENCHANNEL:
            open_channel(network, simulation.random_generator)
        else:
            print("ERROR wrong event type")
            exit(-1)

    if pay_params.mpp:
        post_process_payment_stats(payments)

    time_spent = time.time() - begin
    print(f"Time consumed by simulation events: {time_spent:.2f} s")

    write_output(network, payments, output_dir_name)

if __name__ == "__main__":
    import sys
    main(sys.argv)

