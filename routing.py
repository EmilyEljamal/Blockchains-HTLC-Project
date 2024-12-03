import threading
from math import pow
from typing import List, Optional

from array_ import Array
from heap_ import Heap
from list_ import push, get_by_key, list_len, list_free, is_in_list, pop
from payments import compute_fee
from utils import is_key_equal, is_equal_key_result, is_equal_long
import sys

# Constants
INF = sys.maxsize
N_THREADS = 3
FINALTIMELOCK = 40
HOPSLIMIT = 27
TIMELOCKLIMIT = 2016 + FINALTIMELOCK
PROBABILITYLIMIT = 0.01
PENALTYHALFLIFE = 1
PREVSUCCESSPROBABILITY = 0.95
RISKFACTOR = 15
PAYMENTATTEMPTPENALTY = 100000

# Placeholder global variables
distance_heap: List[Optional[Heap]] = [None] * N_THREADS
data_mutex = threading.Lock()
jobs_mutex = threading.Lock()
paths = []
jobs = None

# Define auxiliary classes and structures (some are placeholders based on header files)
class ThreadArgs:
    def __init__(self, network, payments, current_time, data_index):
        self.network = network
        self.payments = payments
        self.current_time = current_time
        self.data_index = data_index

class Distance:
    def __init__(self, node, distance_=INF, amt_to_receive=0, fee=0, probability=1.0, timelock=0, weight=0, next_edge=-1):
        self.node = node
        self.distance = distance_
        self.amt_to_receive = amt_to_receive
        self.fee = fee
        self.probability = probability
        self.timelock = timelock
        self.weight = weight
        self.next_edge = next_edge

distance: List[Optional[List[Distance]]] = [None] * N_THREADS

class RouteHop:
    def __init__(self, from_node_id, to_node_id, amount_to_forward, timelock, edge_id=None):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.amount_to_forward = amount_to_forward
        self.timelock = timelock
        self.edge_id = edge_id


class Route:
    def __init__(self, total_amount=0, total_fee=0, total_timelock=0):
        self.total_amount = total_amount
        self.total_fee = total_fee
        self.total_timelock = total_timelock
        self.route_hops = Array(5)

class PathHop:
    def __init__(self, sender, receiver, edge):
        self.sender = sender
        self.receiver = receiver
        self.edge = edge


def transform_path_into_route(path, amount, network):
    """
    Transforms a given path into a route for payment.

    :param path: List of node IDs representing the path.
    :param amount: Amount to forward along the path.
    :param network: Network object containing nodes and edges.
    :return: A Route object with hops populated based on the path and network information.
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two nodes.")

    route_hops = []
    timelock = network.final_timelock  # Assuming a predefined constant for the final timelock.
    total_fee = 0

    # Traverse the path and create RouteHops
    for i in range(len(path) - 1):
        from_node_id = path[i]
        to_node_id = path[i + 1]

        # Find the edge connecting these two nodes
        edge = next(
            (edge for edge in network.edges if edge.from_node_id == from_node_id and edge.to_node_id == to_node_id),
            None
        )
        if edge is None:
            raise ValueError(f"No edge found between node {from_node_id} and node {to_node_id} in the network.")

        # Calculate the amount to forward and the fee
        fee = compute_fee(amount, edge.policy)
        amount_to_forward = amount + fee

        # Create a RouteHop
        route_hop = RouteHop(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            edge_id=edge.id_,
            amount_to_forward=amount_to_forward,
            timelock=timelock
        )

        # Update the timelock for the next hop
        timelock += edge.policy.timelock

        # Add the hop to the route
        route_hops.append(route_hop)

        # Update the amount and total fee for the next hop
        amount = amount_to_forward
        total_fee += fee

    # Create and populate the Route object
    route = Route(total_amount=amount, total_fee=total_fee, total_timelock=timelock)
    for hop in route_hops:
        route.route_hops.insert(hop)  # Add each hop to the Route's Array of hops

    return route


# Initialize Dijkstra structures and job queue
def initialize_dijkstra(n_nodes, n_edges, payments):
    global paths, jobs
    paths = [None] * len(payments)
    for i in range(N_THREADS):
        distance[i] = [Distance(node) for node in range(n_nodes)]
        distance_heap[i] = Heap(n_edges)

    jobs = None
    for i in range(len(payments)):
        payment = payments[i]
        jobs = push(jobs, payment.id)

def compare_distance(a, b):
    # Compares two distances: primary by distance, secondary by probability.
    if a.distance == b.distance:
        return -1 if a.probability < b.probability else 1
    return -1 if a.distance < b.distance else 1


class DijkstraThreadArgs:
    def __init__(self, network, payments, current_time, data_index):
        self.network = network
        self.payments = payments
        self.current_time = current_time
        self.data_index = data_index

def run_dijkstra_threads(network, payments, current_time, num_threads=4):
    """
    Runs the Dijkstra algorithm for all payments using multiple threads.
    :param network: The network object containing nodes and edges.
    :param payments: A list of payments for which paths are to be computed.
    :param current_time: The current simulation time.
    :param num_threads: Number of threads to use for parallel computation.
    """
    global jobs, paths

    # Initialize jobs and paths
    jobs = None
    paths = [None] * len(payments)

    # Create thread arguments
    args = DijkstraThreadArgs(network, payments, current_time, data_index=0)

    # Start threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=dijkstra_thread, args=(args,))
        threads.append(thread)
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Return the computed paths
    return paths

def dijkstra_thread(args: DijkstraThreadArgs):
    global jobs, paths
    while True:
        with jobs_mutex:
            if not jobs:  # No jobs left
                return
            jobs, data = pop(jobs)  # Pop from the job queue
            payment_id = data if data else None
        if payment_id is None:
            continue
        with data_mutex:
            payment = args.payments[payment_id]
        hops = dijkstra(payment.sender, payment.receiver, payment.amount, args.network, args.current_time, args.data_index)
        paths[payment_id] = hops


def dijkstra(source, target, amount, network, current_time, p):
    # Initialize the source node and check balance conditions
    source_node = network.nodes.get(source)
    max_balance, total_balance = get_balance(source_node)
    if amount > total_balance:
        return None  # NOLOCALBALANCE error handling
    elif amount > max_balance:
        return None  # NOPATH error handling

    # Clear the distance heap and set initial distances
    while distance_heap[p].length() != 0:
        distance_heap[p].pop(compare_distance)

    for i in range(len(network.nodes)):
        dist = distance[p][i]
        dist.node = i
        dist.distance = INF
        dist.fee = 0
        dist.amt_to_receive = 0
        dist.next_edge = -1

    # Initialize the target node distance
    if distance[p] is None:
        raise ValueError(
            f"distance[{p}] is not initialized. Ensure `initialize_dijkstra` is called before accessing `distance`.")

    distance[p][target].node = target
    distance[p][target].amt_to_receive = amount
    distance[p][target].fee = 0
    distance[p][target].distance = 0
    distance[p][target].timelock = FINALTIMELOCK
    distance[p][target].weight = 0
    distance[p][target].probability = 1

    # Insert target node into the heap
    distance_heap[p].insert(distance[p][target], compare_distance)

    # Main loop: process nodes in the heap
    while distance_heap[p].length() != 0:
        d = distance_heap[p].pop(compare_distance)
        best_node_id = d.node
        if best_node_id == source:
            break

        # Retrieve the distance for the current node
        to_node_dist = distance[p][best_node_id]
        amt_to_send = to_node_dist.amt_to_receive

        # Retrieve best edges for the current node
        best_node = network.nodes.get(best_node_id)
        for edge in best_node.open_edges:
            edge = network.edges.get(edge.counter_edge_id)

            from_node_id = edge.from_node_id
            if from_node_id == source and edge.balance < amt_to_send:
                continue
            elif from_node_id != source:
                channel = network.channels.get(edge.channel_id)
                if channel.capacity < amt_to_send:
                    continue

            if amt_to_send < edge.policy.min_htlc:
                continue

            # Calculate probability for the edge
            edge_probability = get_probability(from_node_id, to_node_dist.node, amt_to_send, source, current_time,
                                               network)
            if edge_probability < PROBABILITYLIMIT:
                continue

            # Calculate fee and timelock for the edge
            edge_fee = compute_fee(amt_to_send, edge.policy) if from_node_id != source else 0
            edge_timelock = edge.policy.timelock if from_node_id != source else 0
            amt_to_receive = amt_to_send + edge_fee

            # Check timelock limits
            tmp_timelock = to_node_dist.timelock + edge_timelock
            if tmp_timelock > TIMELOCKLIMIT:
                continue

            # Calculate the probability and weighted distance
            tmp_probability = to_node_dist.probability * edge_probability
            edge_weight = get_edge_weight(amt_to_receive, edge_fee, edge_timelock)
            tmp_weight = to_node_dist.weight + edge_weight
            tmp_dist = get_probability_based_dist(tmp_weight, tmp_probability)

            # Check if the new distance is better
            current_dist = distance[p][from_node_id].distance
            current_prob = distance[p][from_node_id].probability
            if tmp_dist > current_dist:
                continue
            elif tmp_dist == current_dist and tmp_probability <= current_prob:
                continue

            # Update distance information for the current edge
            dist = distance[p][from_node_id]
            dist.node = from_node_id
            dist.distance = tmp_dist
            dist.weight = tmp_weight
            dist.amt_to_receive = amt_to_receive
            dist.timelock = tmp_timelock
            dist.probability = tmp_probability
            dist.next_edge = edge.id

            # Insert or update the heap with the new distance
            distance_heap[p].insert_or_update(dist, compare_distance, is_key_equal)

    # Convert path into a list of hops if a path was found
    hops = Array(5)
    curr = source
    while curr != target:
        if distance[p][curr].next_edge == -1:
            return None  # NOPATH error handling

        hop = PathHop(sender=curr, edge=distance[p][curr].next_edge,
                      receiver=network.edges.get(distance[p][curr].next_edge).to_node_id)
        hops.insert(hop)
        curr = hop.receiver

    # Check if the number of hops exceeds the limit
    if hops.length() > HOPSLIMIT:
        return None  # NOPATH error handling

    return hops


def millisec_to_hour(time):
    return time / (1000.0 * 60.0 * 60.0)

def get_weight(age):
    exp = -millisec_to_hour(age) / PENALTYHALFLIFE
    return pow(2, exp)

def calculate_probability(node_results, to_node_id, amount, node_probability, current_time):
    result = get_by_key(node_results, to_node_id, is_equal_key_result)
    if not result:
        return node_probability
    if amount <= result.success_amount:
        return PREVSUCCESSPROBABILITY
    if result.fail_time > current_time:
        raise ValueError("ERROR (calculate_probability): fail_time > current_time")
    weight = get_weight(current_time - result.fail_time)
    return node_probability * (1 - weight)

def get_node_probability(node_results, amount, current_time):
    if list_len(node_results) == 0:
        return PREVSUCCESSPROBABILITY
    total_weight = total_probabilities = 0
    for result in node_results:
        if amount <= result.success_amount:
            total_probabilities += PREVSUCCESSPROBABILITY
            total_weight += 1
        elif result.fail_time and amount >= result.fail_amount:
            age = current_time - result.fail_time
            total_weight += get_weight(age)
    return total_probabilities / total_weight if total_weight else PREVSUCCESSPROBABILITY

def get_probability(from_node_id, to_node_id, amount, sender_id, current_time, network):
    sender = network.nodes.get(sender_id)
    node_results = sender.results[from_node_id] if from_node_id != sender_id else None
    if from_node_id == sender_id:
        node_probability = PREVSUCCESSPROBABILITY
    else:
        node_probability = get_node_probability(node_results, amount, current_time)
    return calculate_probability(node_results, to_node_id, amount, node_probability, current_time)

def get_balance(node):
    max_balance = total_balance = 0
    for edge in node.open_edges:
        total_balance += edge.balance
        max_balance = max(max_balance, edge.balance)
    return max_balance, total_balance

def get_best_edges(to_node_id, amount, source_node_id, network):
    best_edges = Array(5)
    explored_nodes = None
    to_node = network.nodes.get(to_node_id)
    for edge in to_node.open_edges:
        if not is_in_list(explored_nodes, edge.to_node_id, is_equal_long):
            explored_nodes = push(explored_nodes, edge.to_node_id)
            best_edge = evaluate_edges(edge, to_node_id, amount, source_node_id, network)
            if best_edge:
                best_edges.insert(best_edge)
    list_free(explored_nodes)
    return best_edges

def evaluate_edges(edge, to_node_id, amount, source_node_id, network):
    from_node_id = edge.to_node_id
    local_node = source_node_id == from_node_id
    best_edge = None
    for candidate_edge in network.nodes.get(to_node_id).open_edges:
        if candidate_edge.to_node_id == from_node_id:
            counter_edge = network.edges.get(candidate_edge.counter_edge_id)
            if local_node:
                if candidate_edge.balance < amount:
                    continue
                best_edge = candidate_edge if best_edge is None else min(best_edge, candidate_edge, key=lambda e: e.balance)
            else:
                if amount < counter_edge.policy.min_htlc or amount > counter_edge.balance:
                    continue
                best_edge = candidate_edge if best_edge is None else min(best_edge, candidate_edge, key=lambda e: e.policy.fee_base)
    return best_edge

def get_edge_weight(amount, fee, timelock):
    timelock_penalty = amount * (timelock * RISKFACTOR) / 1e9
    return timelock_penalty + fee

def get_probability_based_dist(weight, probability):
    min_probability = 0.00001
    return INF if probability < min_probability else weight + PAYMENTATTEMPTPENALTY / probability
