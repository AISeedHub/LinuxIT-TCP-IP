from asyncio import get_event_loop
from tasks import Manager
from util import *


def run_server(cnf):
    loop = get_event_loop()
    manager = Manager(cnf)
    manager.start_server()
    # Run loop
    loop.run_forever()


if __name__ == '__main__':
    # load the configuration
    server_config = load_server_config()
    model_config = load_model_config()
    # combine the two configurations
    config = {**server_config, **model_config}
    run_server(config)
