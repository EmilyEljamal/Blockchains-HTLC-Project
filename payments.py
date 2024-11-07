import random
import math 
import numpy as np # type: ignore
import csv
from dataclasses import dataclass, field

@dataclass
class Payment:
    id: int
    sender: int
    receiver: int
    amount: int
    start_time: int
    route: list = field(default_factory=list)
    is_success: bool = False
    offline_node_count: int = 0
    no_balance_count: int = 0
    is_timeout: bool = False
    end_time: int = 0
    attempts: int = 0
    error_type: str = "NOERROR"
    error_hop: str = None
    is_shard: bool = False
    shards_id: list = field(default_factory=lambda: [-1, -1])

def generate_random_payments(pay_params, n_nodes, random_generator):
    with open("payments.csv", "w", newline='') as payments_file:
        writer = csv.writer(payments_file)
        writer.writerow(["id", "sender_id", "receiver_id", "amount", "start_time"])
        
        payment_time = 1
        for i in range(pay_params.n_payments):
            sender_id = random_generator.randint(0, n_nodes - 1)
            receiver_id = random_generator.randint(0, n_nodes - 1)
            while sender_id == receiver_id:
                receiver_id = random_generator.randint(0, n_nodes - 1)
            
            payment_amount = abs(pay_params.average_amount + random_generator.gauss(0, 1)) * 1000.0
            next_payment_interval = 1000 * random_generator.expovariate(1 / pay_params.inverse_payment_rate)
            payment_time += next_payment_interval
            
            writer.writerow([i, sender_id, receiver_id, int(payment_amount), int(payment_time)])

def generate_payments(pay_params):
    payments = []
    payments_filename = "payments.csv" if not pay_params.payments_from_file else pay_params.payments_filename
    
    with open(payments_filename, "r") as payments_file:
        reader = csv.DictReader(payments_file)
        for row in reader:
            payment = Payment(
                id=int(row['id']),
                sender=int(row['sender_id']),
                receiver=int(row['receiver_id']),
                amount=int(row['amount']),
                start_time=int(row['start_time'])
            )
            payments.append(payment)
    
    return payments

def initialize_payments(pay_params, n_nodes, random_generator):
    if not pay_params.payments_from_file:
        generate_random_payments(pay_params, n_nodes, random_generator)
    return generate_payments(pay_params)

