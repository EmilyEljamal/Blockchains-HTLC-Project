# Blockchains-HTLC-Simulation

This repository implements a **HTLC (Hashed Time-Locked Contracts)** simulation using Python. The project simulates a payment network with routing algorithms, event-driven simulation, and statistical analysis of payment performance. It is inspired by the CLoth simulator which was originally written in C.

---

## Features

- **Network Simulation**: 
  - Nodes, edges, channels, and policies are initialized from configuration files or generated randomly.
  - Open edges and node probabilities are dynamically updated.

- **Routing Algorithms**:
  - Implemented **Dijkstra’s Algorithm** for finding optimal payment paths.
  - Supports multi-hop payment routing.

- **Event Handling**:
  - Event-driven architecture with key events like `FINDPATH`, `SENDPAYMENT`, `FORWARDPAYMENT`, and `RECEIVEPAYMENT`.

- **Payment Analysis**:
  - Statistical outputs for payments, including success rate, route length, attempts, and failure reasons.
  - Batch processing and confidence interval calculations for result consistency.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies: `numpy`, `pandas`, `scipy`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Running the Simulation

1. **Initialize the Network**:
   - Ensure input files (e.g., `nodes_ln.csv`, `channels_ln.csv`, `edges_ln.csv`) are in the `inputs/` directory.
   - Alternatively, use random network generation by modifying `cloth_input.txt`.

2. **Run the Main Simulation**:
   ```bash
   python3 cloth.py outputs/
   ```

3. **Analyze the Results**:
   - Run `batch-means.py` to compute payment statistics:
     ```bash
     python3 batch-means.py outputs/
     ```

---

## Input File Format

- **`cloth_input.txt`**:
  Configuration file for simulation parameters:
  ```
  generate_network_from_file=true
  nodes_filename=inputs/nodes_ln.csv
  channels_filename=inputs/channels_ln.csv
  edges_filename=inputs/edges_ln.csv
  faulty_node_probability=0.0
  n_payments=5000
  average_payment_amount=100
  ```

- **`nodes_ln.csv`**:
  ```
  id
  0
  1
  ...
  ```

- **`channels_ln.csv`**:
  ```
  id,edge1_id,edge2_id,node1_id,node2_id,capacity
  0,0,1,0,1,100000
  ```

- **`edges_ln.csv`**:
  ```
  id,channel_id,counter_edge_id,from_node_id,to_node_id,balance,fee_base,fee_proportional,min_htlc,timelock
  0,0,1,0,1,50000,1000,1,1000,40
  ```

---

## Output Files

1. **`payments_output.csv`**:
   - Contains detailed information on each payment:
     ```
     id,sender_id,receiver_id,amount,start_time,end_time,is_success,no_balance_count,offline_node_count,timeout_exp,attempts,route,total_fee
     0,2567,1551,99661,1,6099,TRUE,0,2,FALSE,3,21524-11369,1000
     ```

2. **`cloth_output.json`**:
   - Summarized simulation statistics:
     ```json
     {
         "Success": {
             "Mean": "0.9500",
             "Variance": "0.0002",
             "ConfidenceMin": "0.9400",
             "ConfidenceMax": "0.9600"
         },
         ...
     }
     ```

---

## Key Functions

- **`initialize_network`** (network.py): Sets up nodes, edges, and channels.
- **`dijkstra`** (routing.py): Finds the optimal payment path.
- **`transform_path_into_route`** (routing.py): Converts a path into a valid payment route.
- **`send_payment`** (htlc.py): Processes a payment across nodes and channels.

---

## Known Issues

- Some runtime errors may occur if input files are not formatted correctly.
- Ensure `timelock` and fee policies align with network capacities.


---