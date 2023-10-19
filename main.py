from asyncio import get_event_loop
from tasks import Manager
from utils import *


def run_server(cnf):
    loop = get_event_loop()
    manager = Manager(cnf)
    manager.start_server()
    # Run loop
    loop.run_forever()


if __name__ == '__main__':
    # load the configuration
    config = load_config()
    run_server(config)
