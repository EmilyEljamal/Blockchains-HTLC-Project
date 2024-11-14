import numpy as np
import sys
import pygsl.rng
import pygsl.randist

from array import *
from event import *
from heap_ import *
from list_ import *
from network import *
from payments import *
from routing import *
from utils import *

OFFLINELATENCY = 3000

# Define the policy structure
class Policy:
    def __init__(self, fee_proportional, fee_base, min_htlc, timelock):
        self.fee_proportional = fee_proportional
        self.fee_base = fee_base
        self.min_htlc = min_htlc
        self.timelock = timelock

# Define the edge structure
class Edge:
    def __init__(self, balance, policy):
        self.balance = balance
        self.policy = policy

# Define the route hop structure
class RouteHop:
    def __init__(self, from_node_id, to_node_id, amount_to_forward, timelock):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.amount_to_forward = amount_to_forward
        self.timelock = timelock

# Define the node pair result structure
class NodePairResult:
    def __init__(self, to_node_id):
        self.to_node_id = to_node_id
        self.fail_time = 0
        self.fail_amount = 0
        self.success_time = 0
        self.success_amount = 0

# Compute the fees to be paid to a hop for forwarding the payment
def compute_fee(amount_to_forward, policy):
    fee = (policy.fee_proportional * amount_to_forward) / 1000000
    return policy.fee_base + fee

# Check whether there is sufficient balance in an edge for forwarding the payment
def check_balance_and_policy(edge, prev_edge, prev_hop, next_hop):
    if next_hop.amount_to_forward > edge.balance:
        return 0

    if next_hop.amount_to_forward < edge.policy.min_htlc:
        print("ERROR: policy.min_htlc not respected", file=sys.stderr)
        sys.exit(-1)

    expected_fee = compute_fee(next_hop.amount_to_forward, edge.policy)
    if prev_hop.amount_to_forward != next_hop.amount_to_forward + expected_fee:
        print("ERROR: policy.fee not respected", file=sys.stderr)
        sys.exit(-1)

    if prev_hop.timelock != next_hop.timelock + prev_edge.policy.timelock:
        print("ERROR: policy.timelock not respected", file=sys.stderr)
        sys.exit(-1)

    return 1

# Retrieve a hop from a payment route
def get_route_hop(node_id, route_hops, is_sender):
    for i in range(route_hops.index):
        route_hop = route_hops.get(i)
        if is_sender and route_hop.from_node_id == node_id:
            index  = i
            break
        if not is_sender and route_hop.to_node_id == node_id:
            index  = i
            break
    if (index == -1):
        return None

    return route_hops.get(index)

# Set the result of a node pair as success
def set_node_pair_result_success(results, from_node_id, to_node_id, success_amount, success_time):
    result = get_by_key(results[from_node_id], to_node_id, is_equal_key_result)

    if result is None:
        result = NodePairResult(to_node_id)
        results[from_node_id] = push(results[from_node_id], result)

    result.success_time = success_time
    if success_amount > result.success_amount:
        result.success_amount = success_amount
    if result.fail_time != 0 and result.success_amount > result.fail_amount:
        result.fail_amount = success_amount + 1

# Set the result of a node pair as fail
def set_node_pair_result_fail(results, from_node_id, to_node_id, fail_amount, fail_time):
    result = get_by_key(results[from_node_id], to_node_id, is_equal_key_result)
    if result is not None:
        if fail_amount > result.fail_amount and fail_time - result.fail_time < 60000:
            return
    if result is None:
        result = NodePairResult(to_node_id)
        results[from_node_id] = push(results[from_node_id], result)
    if fail_amount == 0:
        result.success_amount = 0
    elif fail_amount != 0 and fail_amount <= result.success_amount:
        result.success_amount = fail_amount - 1

# process a payment which succeeded
def process_success_result(node, payment, current_time):
  route_hops = payment.route.route_hops
  for i in range(route_hops):
    hop = route_hops.get(i)
    set_node_pair_result_success(node.results, hop.from_node_id, hop.to_node_id, hop.amount_to_forward, current_time)

# process a payment which failed (different processments depending on the error type) 
def process_fail_result(node, payment, current_time):
  error_hop = payment.error.hop

  if(error_hop.from_node_id == payment.sender) #do nothing if the error was originated by the sender (see `processPaymentOutcomeSelf` in lnd)
    return

  if(payment.error.type == "OFFLINENODE"):
    set_node_pair_result_fail(node.results, error_hop.from_node_id, error_hop.to_node_id, 0, current_time)
    set_node_pair_result_fail(node.results, error_hop.to_node_id, error_hop.from_node_id, 0, current_time)
  elif(payment.error.type == "NOBALANCE"):
    route_hops = payment.route.route_hops
    for i in range(route_hops):
      hop = route_hops.get(i)
      if(hop.edge_id == error_hop.edge_id):
        set_node_pair_result_fail(node.results, hop.from_node_id, hop.to_node_id, hop.amount_to_forward, current_time)
        break
      set_node_pair_result_success(node.results, hop.from_node_id, hop.to_node_id, hop.amount_to_forward, current_time)


def generate_send_payment_event(payment, path, simulation, network):
  route = transform_path_into_route(path, payment.amount, network)
  payment.route = route
  next_event_time = simulation.current_time
  send_payment_event = new_event(next_event_time, "SENDPAYMENT", payment.sender, payment )
  simulation.events = simulation.events.insert(send_payment_event, compare_event)


def create_payment_shard(shard_id, shard_amount, payment):
  shard = new_payment(shard_id, payment.sender, payment.receiver, shard_amount, payment.start_time)
  shard.attempts = 1
  shard.is_shard = 1
  return shard

#HTLC FUNCTIONS

# find a path for a payment (a modified version of dijkstra is used: see `routing.c`) 
def find_path(event, simulation, network, payments, mpp):
  payment = event.payment

  payment.attempts = payment.attempts + 1

  if(simulation.current_time > payment.start_time + 60000):
    payment.end_time = simulation.current_time
    payment.is_timeout = 1
    return

  if (payment.attempts==1):
    path = paths[payment.id]
  else:
    path = dijkstra(payment.sender, payment.receiver, payment.amount, network, simulation.current_time, 0)

  if (path != None):
    generate_send_payment_event(payment, path, simulation, network)
    return
 

  # if a path is not found, try to split the payment in two shards (multi-path payment)
  if (mpp and path == None and (not payment.is_shard) and payment.attempts == 1):
    shard1_amount = payment.amount/2
    shard2_amount = payment.amount - shard1_amount
    shard1_path = dijkstra(payment.sender, payment.receiver, shard1_amount, network, simulation.current_time, 0)
    if(shard1_path == None):
      payment.end_time = simulation.current_time
      return
   
    shard2_path = dijkstra(payment.sender, payment.receiver, shard2_amount, network, simulation.current_time, 0)
    if(shard2_path == None):
      payment.end_time = simulation.current_time
      return
   
    shard1_id = payments.index
    shard2_id = payments.index + 1
    shard1 = create_payment_shard(shard1_id, shard1_amount, payment)
    shard2 = create_payment_shard(shard2_id, shard2_amount, payment)
    payments = payments.insert(shard1)
    payments = payments.insert(shard2)
    payment.is_shard = 1
    payment.shards_id[0] = shard1_id
    payment.shards_id[1] = shard2_id
    generate_send_payment_event(shard1, shard1_path, simulation, network)
    generate_send_payment_event(shard2, shard2_path, simulation, network)
    return
 
  payment.end_time = simulation.current_time
  return

# send an HTLC for the payment (behavior of the payment sender) 
def send_payment(event, simulation, network):
  payment = event.payment
  route = payment.route
  node = network.nodes.get(event.node_id)
  first_route_hop = route.route_hops.get(0)
  next_edge = network.edges.get(first_route_hop.edge_id)

  if(not is_present(next_edge.id, node.open_edges)):
    print("ERROR (send_payment): edge %ld is not an edge of node %ld \n", next_edge.id, node.id)
    sys.exit(-1)
 

  # simulate the case that the next node in the route is offline 
  is_next_node_offline = gsl_ran_discrete(simulation.random_generator, network.faulty_node_prob)
  if(is_next_node_offline):
    payment.offline_node_count += 1
    payment.error.type = "OFFLINENODE"
    payment.error.hop = first_route_hop
    next_event_time = simulation.current_time + OFFLINELATENCY
    next_event = new_event(next_event_time, "RECEIVEFAIL", event.node_id, event.payment)
    simulation.events = simulation.events.insert(next_event, compare_event)
    return
 

  if(first_route_hop.amount_to_forward > next_edge.balance):
    payment.error.type = "NOBALANCE"
    payment.error.hop = first_route_hop
    payment.no_balance_count += 1
    next_event_time = simulation.current_time
    next_event = new_event(next_event_time, "RECEIVEFAIL", event.node_id, event.payment )
    simulation.events = simulation.events.insert(next_event, compare_event)
    return
 

  next_edge.balance -= first_route_hop.amount_to_forward

  next_edge.tot_flows += 1

  event_type = "RECEIVEPAYMENT" if first_route_hop.to_node_id == payment.receiver else "FORWARDPAYMENT"
  next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator)
  next_event = new_event(next_event_time, event_type, first_route_hop.to_node_id, event.payment)
  simulation.events = simulation.events.insert(next_event, compare_event)


# forward an HTLC for the payment (behavior of an intermediate hop node in a route) 
def forward_payment(event, simulation, network):
  next_edge = None, prev_edge

  payment = event.payment
  node = network.nodes.get(event.node_id)
  route = payment.route
  next_route_hop=get_route_hop(node.id, route.route_hops, 1)
  previous_route_hop = get_route_hop(node.id, route.route_hops, 0)
  is_last_hop = next_route_hop.to_node_id == payment.receiver

  if(not is_present(next_route_hop.edge_id, node.open_edges)):
    print("ERROR (forward_payment): edge %ld is not an edge of node %ld \n", next_route_hop.edge_id, node.id)
    sys.exit(-1)
 

  # simulate the case that the next node in the route is offline 
  is_next_node_offline = gsl_ran_discrete(simulation.random_generator, network.faulty_node_prob)
  if(is_next_node_offline and not is_last_hop): #assume that the receiver node is always online
    payment.offline_node_count += 1
    payment.error.type = "OFFLINENODE"
    payment.error.hop = next_route_hop
    prev_node_id = previous_route_hop.from_node_id
    event_type = 'RECEIVEFAIL' if prev_node_id == payment.sender else 'FORWARDFAIL'
    next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator) + OFFLINELATENCY
    next_event = new_event(next_event_time, event_type, prev_node_id, event.payment)
    simulation.events = simulation.events.insert(next_event, compare_event)
    return

  # STRICT FORWARDING 
  prev_edge = network.edges.get(previous_route_hop.edge_id)
  next_edge = network.edges.get(next_route_hop.edge_id)
  can_send_htlc = check_balance_and_policy(next_edge, prev_edge, previous_route_hop, next_route_hop)
  if(not can_send_htlc):
    payment.error.type = "NOBALANCE"
    payment.error.hop = next_route_hop
    payment.no_balance_count += 1
    prev_node_id = previous_route_hop.from_node_id
    event_type = 'RECEIVEFAIL' if prev_node_id == payment.sender else 'FORWARDFAIL'
    next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator) #prev_channel.latency
    next_event = new_event(next_event_time, event_type, prev_node_id, event.payment)
    simulation.events = simulation.events.insert(next_event, compare_event)
    return
 

  next_edge.balance -= next_route_hop.amount_to_forward

  next_edge.tot_flows += 1

  event_type = "RECEIVEPAYMENT" if is_last_hop else "FORWARDPAYMENT"
  next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator) #next_channel.latency
  next_event = new_event(next_event_time, event_type, next_route_hop.to_node_id, event.payment)
  simulation.events = simulation.events.insert(next_event, compare_event)
}

# receive a payment (behavior of the payment receiver node) 
def receive_payment(event,simulation, network):
  payment = event.payment
  route = payment.route
  node = network.nodes.get(event.node_id)

  last_route_hop = route.route_hops.get(route.route_hops.index - 1)
  forward_edge = network.edges.get(last_route_hop.edge_id)
  backward_edge = network.edges.get(forward_edge.counter_edge_id)

  if(not is_present(backward_edge.id, node.open_edges)):
    print("ERROR (receive_payment): edge %ld is not an edge of node %ld \n", backward_edge.id, node.id)
    sys.exit(-1)
 

  backward_edge.balance += last_route_hop.amount_to_forward

  payment.is_success = 1

  prev_node_id = last_route_hop.from_node_id
  event_type = 'RECEIVESUCCESS' if prev_node_id == payment.sender else 'FORWARDSUCCESS'
  next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator) #channel.latency
  next_event = new_event(next_event_time, event_type, prev_node_id, event.payment)
  simulation.events = simulation.events.insert(next_event, compare_event)

# forward an HTLC success back to the payment sender (behavior of a intermediate hop node in the route) 
def forward_success(event, simulation, network):
  payment = event.payment
  prev_hop = get_route_hop(event.node_id, payment.route.route_hops, 0)
  forward_edge = network.edges.get(prev_hop.edge_id)
  backward_edge = network.edges.get(forward_edge.counter_edge_id)
  node = network.nodes.get(event.node_id)

  if(not is_present(backward_edge.id, node.open_edges)):
    print("ERROR (forward_success): edge %ld is not an edge of node %ld \n", backward_edge.id, node.id)
    sys.exit(-1)
 

  backward_edge.balance += prev_hop.amount_to_forward

  prev_node_id = prev_hop.from_node_id
  event_type = 'RECEIVESUCCESS' if prev_node_id == payment.sender else 'FORWARDSUCCESS'
  next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator) # prev_channel.latency
  next_event = new_event(next_event_time, event_type, prev_node_id, event.payment)
  simulation.events = simulation.events.insert(next_event, compare_event)

# receive an HTLC success (behavior of the payment sender node) 
def receive_success(event, simulation, network):
  payment = event.payment
  node = network.nodes.get(event.node_id)
  event.payment.end_time = simulation.current_time
  process_success_result(node, payment, simulation.current_time)

# forward an HTLC fail back to the payment sender (behavior of a intermediate hop node in the route) 
def forward_fail(event, simulation, network):
  node = network.nodes.get(event.node_id)
  payment = event.payment
  next_hop = get_route_hop(event.node_id, payment.route.route_hops, 1)
  next_edge = network.edges.get(next_hop.edge_id)

  if(not is_present(next_edge.id, node.open_edges)):
    print("ERROR (forward_fail): edge %ld is not an edge of node %ld \n", next_edge.id, node.id)
    sys.exit(-1)
 

  # since the payment failed, the balance must be brought back to the state before the payment occurred  
  next_edge.balance += next_hop.amount_to_forward

  prev_hop = get_route_hop(event.node_id, payment.route.route_hops, 0)
  prev_node_id = prev_hop.from_node_id
  event_type = 'RECEIVEFAIL' if prev_node_id == payment.sender else 'FORWARDFAIL'
  next_event_time = simulation.current_time + 100 + gsl_ran_ugaussian(simulation.random_generator) # prev_channel.latency
  next_event = new_event(next_event_time, event_type, prev_node_id, event.payment)
  simulation.events = simulation.events.insert(next_event, compare_event)


# receive an HTLC fail (behavior of the payment sender node) 
def receive_fail(event, simulation, network):
  payment = event.payment
  node = network.nodes.get(event.node_id)

  error_hop = payment.error.hop
  if(error_hop.from_node_id != payment.sender): # if the error occurred in the first hop, the balance hasn't to be updated, since it was not decreased
    first_hop = payment.route.route_hops.get(0)
    next_edge = network.edges.get(first_hop.edge_id)
    if(not is_present(next_edge.id, node.open_edges)):
      print("ERROR (receive_fail): edge %ld is not an edge of node %ld \n", next_edge.id, node.id)
      sys.exit(-1)
   
    next_edge.balance += first_hop.amount_to_forward
 

  process_fail_result(node, payment, simulation.current_time)

  next_event_time = simulation.current_time
  next_event = new_event(next_event_time, "FINDPATH", payment.sender, payment)
  simulation.events = simulation.events.insert(next_event, compare_event)

