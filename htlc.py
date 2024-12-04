import numpy as np
import sys
from array_ import Array
from list_ import Element, get_by_key, push
from event import new_event, compare_event
import event as ev
from network import Network, Edge, Node, Policy, Channel
from payments import Payment, compute_fee
import payments as ps
from routing import dijkstra, transform_path_into_route
from utils import is_present, is_equal_key_result
from typing import List, Optional

from logger import print_

OFFLINELATENCY = 3000  # 3 seconds

# Define the route hop structure
class RouteHop:
    def __init__(self, from_node_id: int, to_node_id: int, amount_to_forward: int, timelock: int):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.amount_to_forward = amount_to_forward
        self.timelock = timelock


# Define the node pair result structure
class NodePairResult:
    def __init__(self, to_node_id: int):
        self.to_node_id = to_node_id
        self.fail_time = 0
        self.fail_amount = 0
        self.success_time = 0
        self.success_amount = 0


def open_channel(network, random_generator):
    """
    Opens a new payment channel during the simulation.
    """
    channel_id = len(network.channels)
    edge1_id = len(network.edges)
    edge2_id = edge1_id + 1

    node1_id = random_generator.integers(0, len(network.nodes))
    node2_id = random_generator.integers(0, len(network.nodes))

    while node2_id == node1_id:
        node2_id = random_generator.integers(0, len(network.nodes))

    # Generate a random capacity for the channel
    capacity = random_generator.integers(1_000_000, 10_000_000)  # Example range in millisatoshis

    # Create policies for the edges
    policy1 = Policy(
        fee_proportional=random_generator.integers(1, 10),
        fee_base=random_generator.integers(1000, 5000),
        min_htlc=random_generator.integers(1, 1000),
        timelock=random_generator.integers(10, 100)
    )
    policy2 = Policy(
        fee_proportional=random_generator.integers(1, 10),
        fee_base=random_generator.integers(1000, 5000),
        min_htlc=random_generator.integers(1, 1000),
        timelock=random_generator.integers(10, 100)
    )

    # Create edges with additional parameters
    edge1 = Edge(
        id_=edge1_id,
        channel_id=channel_id,
        counter_edge_id=edge2_id,
        from_node_id=node1_id,
        to_node_id=node2_id,
        balance=capacity // 2,
        policy=policy1
    )
    edge2 = Edge(
        id_=edge2_id,
        channel_id=channel_id,
        counter_edge_id=edge1_id,
        from_node_id=node2_id,
        to_node_id=node1_id,
        balance=capacity // 2,
        policy=policy2
    )

    # Create the channel
    channel = Channel(channel_id, edge1_id, edge2_id, node1_id, node2_id, capacity)

    # Add channel and edges to the network
    network.channels.append(channel)
    network.edges.append(edge1)
    network.edges.append(edge2)

    # Update nodes
    network.nodes.get(node1_id).open_edges.append(edge1_id)
    network.nodes.get(node2_id).open_edges.append(edge2_id)



# Check whether there is sufficient balance in an edge for forwarding the payment
def check_balance_and_policy(edge: Edge, prev_edge: Edge, prev_hop: RouteHop, next_hop: RouteHop) -> bool:
    if next_hop.amount_to_forward > edge.balance:
        return False

    if next_hop.amount_to_forward < edge.policy.min_htlc:
        print("ERROR: policy.min_htlc not respected", file=sys.stderr)
        return False

    expected_fee = compute_fee(next_hop.amount_to_forward, edge.policy)
    if prev_hop.amount_to_forward + expected_fee != next_hop.amount_to_forward:
        print("ERROR: policy.fee not respected", file=sys.stderr)
        return False

    if prev_hop.timelock + prev_edge.policy.timelock != next_hop.timelock:
        print("ERROR: policy.timelock not respected", file=sys.stderr)
        return False

    return True


# Retrieve a hop from a payment route
def get_route_hop(node_id, route_hops, is_sender):
    for route_hop in route_hops:
        if is_sender and route_hop.from_node_id == node_id:
            return route_hop
        if not is_sender and route_hop.to_node_id == node_id:
            return route_hop
    return None



# Set the result of a node pair as success
def set_node_pair_result_success(results: Optional[List[Optional[Element]]], from_node_id: int, to_node_id: int, success_amount: int, success_time: int):
    if results is None:
        raise ValueError("results must be initialized before use.")

        # Ensure results[from_node_id] exists
    if results[from_node_id] is None:
        results[from_node_id] = None  # Initialize as an empty list wrapped in an `Element`

        # Retrieve or create the list of `NodePairResult` for this node
    node_pair_results: Element = results[from_node_id].data if isinstance(results[from_node_id],
                                                                                        Element) else None

    # Find the result for `to_node_id`, or None
    result = get_by_key(node_pair_results, to_node_id, is_equal_key_result)

    # If no result exists, create a new one and add it to the list
    if result is None:
        result = NodePairResult(to_node_id)
        push(node_pair_results, result)

    # Update the result with success details
    result.success_time = success_time
    if success_amount > result.success_amount:
        result.success_amount = success_amount
    if result.fail_time != 0 and result.success_amount > result.fail_amount:
        result.fail_amount = success_amount + 1

    results[from_node_id] = node_pair_results


# Set the result of a node pair as fail
def set_node_pair_result_fail(results: Optional[List[Optional[Element]]], from_node_id: int, to_node_id: int, fail_amount: int, fail_time: int):
    if results is None:
        raise ValueError("results must be initialized before use.")

        # Ensure results[from_node_id] exists
    if results[from_node_id] is None:
        results[from_node_id] = None  # Initialize as an empty list wrapped in an `Element`

        # Retrieve or create the list of `NodePairResult` for this node
    node_pair_results: Element = results[from_node_id].data if isinstance(results[from_node_id],
                                                                                        Element) else None

    # Find the result for `to_node_id`, or None
    result = get_by_key(node_pair_results, to_node_id, is_equal_key_result)

    # If no result exists, create a new one and add it to the list
    if result is None:
        result = NodePairResult(to_node_id)
        push(node_pair_results, result)

    # Check the conditions for updating the result
    if fail_amount > result.fail_amount and fail_time - result.fail_time < 60000:
        return

    # Update the result with fail details
    result.fail_amount = fail_amount
    result.fail_time = fail_time
    if fail_amount == 0:
        result.success_amount = 0
    elif fail_amount != 0 and fail_amount <= result.success_amount:
        result.success_amount = fail_amount - 1

    # Re-wrap the list back into the `Element` if needed
    results[from_node_id] = node_pair_results


# Process a payment which succeeded
def process_success_result(node: Node, payment: Payment, current_time: int):
    for hop in payment.route.route_hops:
        set_node_pair_result_success(node.results, hop.from_node_id, hop.to_node_id, hop.amount_to_forward, current_time)


# Process a payment which failed
def process_fail_result(node: Node, payment: Payment, current_time: int):
    error_hop = payment.error.hop

    if error_hop.from_node_id == payment.sender:
        return

    if payment.error.type == ps.PaymentErrorType.OFFLINENODE:
        set_node_pair_result_fail(node.results, error_hop.from_node_id, error_hop.to_node_id, 0, current_time)
        set_node_pair_result_fail(node.results, error_hop.to_node_id, error_hop.from_node_id, 0, current_time)
    elif payment.error.type == ps.PaymentErrorType.NOBALANCE:
        for hop in payment.route.route_hops:
            if hop.edge_id == error_hop.edge_id:
                set_node_pair_result_fail(node.results, hop.from_node_id, hop.to_node_id, hop.amount_to_forward, current_time)
                break
            set_node_pair_result_success(node.results, hop.from_node_id, hop.to_node_id, hop.amount_to_forward, current_time)


# Generate a send payment event
def generate_send_payment_event(payment: Payment, path: Array, simulation, network: Network):
    route = transform_path_into_route(path, payment.amount, network)
    payment.route = route
    next_event_time = simulation.current_time
    send_payment_event = new_event(next_event_time, ev.EventType.SENDPAYMENT, payment.sender, payment)
    simulation.events.insert(send_payment_event, compare_event)


# Create a payment shard
def create_payment_shard(shard_id: int, shard_amount: int, payment: Payment) -> Payment:
    shard = Payment(
        id_=shard_id,
        sender=payment.sender,
        receiver=payment.receiver,
        amount=shard_amount,
        start_time=payment.start_time
    )
    shard.attempts = 1
    shard.is_shard = True
    return shard

paths = {}

def initialize_paths(payments, network):
    """
    Initialize the `paths` dictionary with precomputed paths for all payments.
    Args:
        payments: List of Payment objects.
        network: Network object to calculate paths.
    """
    global paths
    for payment in payments:
        # Compute the initial path for each payment using dijkstra or another pathfinding algorithm
        path = dijkstra(payment.sender, payment.receiver, payment.amount, network, 0, 0)
        if path is not None:
            print_("Found")
        else:
            print_("Not found")
        paths[payment.id] = path

# Find a path for a payment
def find_path(event, simulation, network: Network, payments: List[Payment], mpp: bool):
    global paths
    payment = event.payment
    payment.attempts += 1

    if simulation.current_time > payment.start_time + 60000:
        payment.end_time = simulation.current_time
        payment.is_timeout = True
        return

    if payment.attempts == 1:
        path = paths.get(payment.id)
    else:
        path = dijkstra(payment.sender, payment.receiver, payment.amount, network, simulation.current_time, 0)

    if path:
        generate_send_payment_event(payment, path, simulation, network)
        return

    # Multi-Path Payment (MPP) Logic
    if mpp and not path and not payment.is_shard and payment.attempts == 1:
        shard1_amount = payment.amount // 2
        shard2_amount = payment.amount - shard1_amount

        shard1_path = dijkstra(payment.sender, payment.receiver, shard1_amount, network, simulation.current_time, 0)
        if not shard1_path:
            payment.end_time = simulation.current_time
            return

        shard2_path = dijkstra(payment.sender, payment.receiver, shard2_amount, network, simulation.current_time, 0)
        if not shard2_path:
            payment.end_time = simulation.current_time
            return

        shard1_id = len(payments)
        shard2_id = len(payments) + 1
        shard1 = create_payment_shard(shard1_id, shard1_amount, payment)
        shard2 = create_payment_shard(shard2_id, shard2_amount, payment)

        payments.append(shard1)
        payments.append(shard2)

        payment.is_shard = True
        payment.shards_id = [shard1_id, shard2_id]

        generate_send_payment_event(shard1, shard1_path, simulation, network)
        generate_send_payment_event(shard2, shard2_path, simulation, network)
        return

    payment.end_time = simulation.current_time


# Send an HTLC for the payment
def send_payment(event, simulation, network: Network):
    payment = event.payment
    route = payment.route
    node = network.nodes.get(event.node_id)
    first_route_hop = route.route_hops.get(0)
    next_edge = network.edges.get(first_route_hop.edge_id)

    if not is_present(next_edge.id, node.open_edges):
        print(f"ERROR (send_payment): edge {next_edge.id} is not an edge of node {node.id}", file=sys.stderr)
        sys.exit(-1)

    is_next_node_offline = np.random.choice([0, 1], p=network.faulty_node_prob)
    if is_next_node_offline:
        payment.offline_node_count += 1
        payment.error.type = ps.PaymentErrorType.OFFLINENODE
        payment.error.hop = first_route_hop
        next_event_time = simulation.current_time + OFFLINELATENCY
        next_event = new_event(next_event_time, ev.EventType.RECEIVEFAIL, event.node_id, payment)
        simulation.events.insert(next_event, compare_event)
        return

    if first_route_hop.amount_to_forward > next_edge.balance:
        payment.error.type = ps.PaymentErrorType.NOBALANCE
        payment.error.hop = first_route_hop
        payment.no_balance_count += 1
        next_event_time = simulation.current_time
        next_event = new_event(next_event_time, ev.EventType.RECEIVEFAIL, event.node_id, payment)
        simulation.events.insert(next_event, compare_event)
        return

    next_edge.balance -= first_route_hop.amount_to_forward
    next_edge.tot_flows += 1

    event_type = ev.EventType.RECEIVEPAYMENT if first_route_hop.to_node_id == payment.receiver else ev.EventType.FORWARDPAYMENT
    next_event_time = simulation.current_time + 100 + np.random.normal()
    next_event = new_event(next_event_time, event_type, first_route_hop.to_node_id, payment)
    simulation.events.insert(next_event, compare_event)


# Forward an HTLC for the payment
def forward_payment(event, simulation, network: Network):
    payment = event.payment
    node = network.nodes.get(event.node_id)
    route = payment.route

    next_route_hop = get_route_hop(node.id, route.route_hops, True)
    previous_route_hop = get_route_hop(node.id, route.route_hops, False)

    is_last_hop = next_route_hop.to_node_id == payment.receiver

    if not is_present(next_route_hop.edge_id, node.open_edges):
        print(f"ERROR (forward_payment): edge {next_route_hop.edge_id} is not an edge of node {node.id}", file=sys.stderr)
        sys.exit(-1)

    is_next_node_offline = np.random.choice([0, 1], p=network.faulty_node_prob)
    if is_next_node_offline and not is_last_hop:
        payment.offline_node_count += 1
        payment.error.type = ps.PaymentErrorType.OFFLINENODE
        payment.error.hop = next_route_hop
        prev_node_id = previous_route_hop.from_node_id
        event_type = ev.EventType.RECEIVEFAIL if prev_node_id == payment.sender else ev.EventType.FORWARDFAIL
        next_event_time = simulation.current_time + OFFLINELATENCY + np.random.normal()
        next_event = new_event(next_event_time, event_type, prev_node_id, payment)
        simulation.events.insert(next_event, compare_event)
        return

    prev_edge = network.edges.get(previous_route_hop.edge_id)
    next_edge = network.edges.get(next_route_hop.edge_id)

    can_forward = check_balance_and_policy(next_edge, prev_edge, previous_route_hop, next_route_hop)
    if not can_forward:
        payment.error.type = ps.PaymentErrorType.NOBALANCE
        payment.error.hop = next_route_hop
        payment.no_balance_count += 1
        prev_node_id = previous_route_hop.from_node_id
        event_type = ev.EventType.RECEIVEFAIL if prev_node_id == payment.sender else ev.EventType.FORWARDFAIL
        next_event_time = simulation.current_time + np.random.normal()
        next_event = new_event(next_event_time, event_type, prev_node_id, payment)
        simulation.events.insert(next_event, compare_event)
        return

    next_edge.balance -= next_route_hop.amount_to_forward
    next_edge.tot_flows += 1

    event_type = ev.EventType.RECEIVEPAYMENT if is_last_hop else ev.EventType.FORWARDPAYMENT
    next_event_time = simulation.current_time + np.random.normal()
    next_event = new_event(next_event_time, event_type, next_route_hop.to_node_id, payment)
    simulation.events.insert(next_event, compare_event)


# Receive a payment
def receive_payment(event, simulation, network: Network):
    payment = event.payment
    route = payment.route
    node = network.nodes.get(event.node_id)

    last_route_hop = route.route_hops.get(-1)
    forward_edge = network.edges.get(last_route_hop.edge_id)
    backward_edge = network.edges.get(forward_edge.counter_edge_id)

    if not is_present(backward_edge.id, node.open_edges):
        print(f"ERROR (receive_payment): edge {backward_edge.id} is not an edge of node {node.id}", file=sys.stderr)
        sys.exit(-1)

    backward_edge.balance += last_route_hop.amount_to_forward
    payment.is_success = True

    prev_node_id = last_route_hop.from_node_id
    event_type = ev.EventType.RECEIVESUCCESS if prev_node_id == payment.sender else ev.EventType.FORWARDSUCCESS
    next_event_time = simulation.current_time + np.random.normal()
    next_event = new_event(next_event_time, event_type, prev_node_id, payment)
    simulation.events.insert(next_event, compare_event)


# Forward an HTLC success back to the payment sender (intermediate hop node behavior)
def forward_success(event, simulation, network: Network):
    payment = event.payment
    prev_hop = get_route_hop(event.node_id, payment.route.route_hops, False)
    forward_edge = network.edges.get(prev_hop.edge_id)
    backward_edge = network.edges.get(forward_edge.counter_edge_id)
    node = network.nodes.get(event.node_id)

    if not is_present(backward_edge.id, node.open_edges):
        print(f"ERROR (forward_success): edge {backward_edge.id} is not an edge of node {node.id}", file=sys.stderr)
        sys.exit(-1)

    backward_edge.balance += prev_hop.amount_to_forward

    prev_node_id = prev_hop.from_node_id
    event_type = ev.EventType.RECEIVESUCCESS if prev_node_id == payment.sender else ev.EventType.FORWARDSUCCESS
    next_event_time = simulation.current_time + np.random.normal()
    next_event = new_event(next_event_time, event_type, prev_node_id, payment)
    simulation.events.insert(next_event, compare_event)


# Receive an HTLC success (payment sender behavior)
def receive_success(event, simulation, network: Network):
    payment = event.payment
    node = network.nodes.get(event.node_id)
    payment.end_time = simulation.current_time
    process_success_result(node, payment, simulation.current_time)


# Forward an HTLC fail back to the payment sender (intermediate hop node behavior)
def forward_fail(event, simulation, network: Network):
    node = network.nodes.get(event.node_id)
    payment = event.payment
    next_hop = get_route_hop(event.node_id, payment.route.route_hops, True)
    next_edge = network.edges.get(next_hop.edge_id)

    if not is_present(next_edge.id, node.open_edges):
        print(f"ERROR (forward_fail): edge {next_edge.id} is not an edge of node {node.id}", file=sys.stderr)
        sys.exit(-1)

    # Restore the balance to the state before the payment occurred
    next_edge.balance += next_hop.amount_to_forward

    prev_hop = get_route_hop(event.node_id, payment.route.route_hops, False)
    prev_node_id = prev_hop.from_node_id
    event_type = ev.EventType.RECEIVEFAIL if prev_node_id == payment.sender else ev.EventType.FORWARDFAIL
    next_event_time = simulation.current_time + np.random.normal()
    next_event = new_event(next_event_time, event_type, prev_node_id, payment)
    simulation.events.insert(next_event, compare_event)


# Receive an HTLC fail (payment sender behavior)
def receive_fail(event, simulation, network: Network):
    payment = event.payment
    node = network.nodes.get(event.node_id)

    error_hop = payment.error.hop
    if error_hop.from_node_id != payment.sender:  # Update balance if the failure is not in the first hop
        first_hop = payment.route.route_hops.get(0)
        next_edge = network.edges.get(first_hop.edge_id)

        if not is_present(next_edge.id, node.open_edges):
            print(f"ERROR (receive_fail): edge {next_edge.id} is not an edge of node {node.id}", file=sys.stderr)
            sys.exit(-1)

        next_edge.balance += first_hop.amount_to_forward

    process_fail_result(node, payment, simulation.current_time)

    next_event_time = simulation.current_time
    next_event = new_event(next_event_time, ev.EventType.FINDPATH, payment.sender, payment)
    simulation.events.insert(next_event, compare_event)
