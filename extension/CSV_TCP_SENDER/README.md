# CSV_TCP_SENDER
* Author : Heonzoo
* version : Python 3.10.18

# Introduction
Linux_IT will develop gateway for sending environmental data to goverment server. So we are trying to send data from our sensor node to gateway.

# Dependencies
* uv - [get-started](https://docs.astral.sh/uv/getting-started/installation/)


# How to Run
1. Open `config.yaml` and Edit IP and Port for gateway. (If no gateway exist, use `virtual_gateway.py` for temporal test. run `python virtual_gateway.py`)
1. Also, edit csv file path from AgroTrack
1. (Optional) change `send_interval_seconds: 20 # seconds` to 600 seconds
1. run `python csv_tcp_sender.py`