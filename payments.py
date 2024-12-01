import numpy as np
import csv
from typing import List
from enum import Enum

# Define PaymentErrorType
class PaymentErrorType(Enum):
    NOERROR = "NOERROR"
    NOBALANCE = "NOBALANCE"
    OFFLINENODE = "OFFLINENODE"

class Payment:
    def __init__(self, id_, sender, receiver, amount, start_time):
        self.id = id_
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.start_time = start_time
        self.route = None
        self.is_success = False
        self.offline_node_count = 0
        self.no_balance_count = 0
        self.is_timeout = False
        self.end_time = 0
        self.attempts = 0
        self.error = Error(type_="NOERROR", hop=None)

class Error:
    def __init__(self, type_, hop):
        self.type = type_
        self.hop = hop


# Create a new Payment
def new_payment(id_: int, sender: int, receiver: int, amount: int, start_time: int) -> Payment:
    return Payment(
        id_=id_,
        sender=sender,
        receiver=receiver,
        amount=amount,
        start_time=start_time
    )

# Generate Random Payments
def generate_random_payments(pay_params, n_nodes: int, random_generator: np.random.Generator) -> None:
    payments_filename = "payments.csv"
    try:
        with open(payments_filename, "w", newline='') as payments_file:
            writer = csv.writer(payments_file)
            writer.writerow(["id", "sender_id", "receiver_id", "amount", "start_time"])

            payment_time = 1
            for i in range(pay_params.n_payments):
                sender_id = random_generator.integers(0, n_nodes)
                receiver_id = random_generator.integers(0, n_nodes)
                while sender_id == receiver_id:
                    receiver_id = random_generator.integers(0, n_nodes)

                payment_amount = int(abs(pay_params.average_amount + random_generator.normal(0, 1)) * 1000)
                next_payment_interval = int(1000 * random_generator.exponential(scale=pay_params.inverse_payment_rate))
                payment_time += next_payment_interval

                writer.writerow([i, sender_id, receiver_id, payment_amount, payment_time])
    except Exception as e:
        raise RuntimeError(f"Error generating random payments: {e}")

# Generate Payments from File
def generate_payments(pay_params) -> List[Payment]:
    """Reads payments from a CSV file and returns a list of Payment objects."""
    payments = []
    payments_filename = "payments.csv" if not pay_params.payments_from_file else pay_params.payments_filename

    try:
        with open(payments_filename, "r") as payments_file:
            reader = csv.reader(payments_file)  # Changed to basic reader
            headers = next(reader)  # Extract headers
            for row in reader:
                # Map row values to header keys
                row_dict = dict(zip(headers, row))
                payment = Payment(
                    id_=int(row_dict['id']),
                    sender=int(row_dict['sender_id']),
                    receiver=int(row_dict['receiver_id']),
                    amount=int(row_dict['amount']),
                    start_time=int(row_dict['start_time'])
                )
                payments.append(payment)
    except FileNotFoundError:
        raise RuntimeError(f"File '{payments_filename}' not found. Please ensure it exists or generate payments first.")
    except ValueError as ve:
        raise RuntimeError(f"Error parsing payments file '{payments_filename}': {ve}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading payments file '{payments_filename}': {e}")

    return payments


# Initialize Payments
def initialize_payments(pay_params, n_nodes: int, random_generator: np.random.Generator) -> List[Payment]:
    if not pay_params.payments_from_file:
        generate_random_payments(pay_params, n_nodes, random_generator)
    return generate_payments(pay_params)
