from socket import socket, SO_REUSEADDR, SOL_SOCKET
from asyncio import Task, coroutine
import asyncio


class Peer(object):
    def __init__(self, server, sock, name):
        self.loop = server.loop
        self._taskManager = server._taskManager
        self.name = name
        self._sock = sock
        self._server = server
        self.task = Task(self._peer_handler())

    def send(self, data):
        # return self.loop.sock_sendall(self._sock, data.encode(ENCODING))
        return self.loop.sock_sendall(self._sock, data.encode(self._server._config["ENCODING"]))

    @coroutine
    def _peer_handler(self):
        try:
            # await self._peer_loop()
            yield from self._peer_loop()
        except IOError:
            pass
        finally:
            self._server.remove(self)

    # @coroutine
    async def _peer_loop(self):
        while True:
            buf = await self.loop.sock_recv(self._sock, self._server._config["BUFFER_SIZE"])
            if buf == b'':
                break

            data = buf.decode(self._server._config["ENCODING"])
            message = 'Receive data from %s: %s' % (self.name, data)
            print(message)

            json_response = self._taskManager.distribute_task(data)
            print("json_response: ", json_response)
            # Return the message to the client
            await self._server.broadcast(json_response)



class Server(object):
    def __init__(self, manager):
        self._taskManager = manager
        self.loop = manager.loop  # Event loops schedules and manages tasks
        self._config = manager.config
        self._serv_sock = socket()
        self._serv_sock.setblocking(0)
        self._serv_sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self._serv_sock.bind(('', self._config["PORT"]))
        self._serv_sock.listen(self._config["MAX_CONNECTIONS"])
        self._peers = []

    def start(self):
        print("Server Address:", self._config["IP"] + ":" + str(self._config["PORT"]))
        print("Start the server...")
        return Task(self._server(), name="server")

    def start_input(self):
        print("Start input listener...")
        return Task(self.input_peers_info(), name="input_peers_info")

    def remove(self, peer):
        self._peers.remove(peer)
        print("Bye bye %s" % peer.name)
        self.broadcast('Peer %s quit!\n' % peer.name)

    def broadcast(self, message):
        print("Broadcasting message...")
        for peer in self._peers:
            print(f'Send message "{message}" to {peer.name}')
            peer._sock.sendall(message.encode(self._config["ENCODING"]))
            # peer.send(message)
            print("Done sending")


    @coroutine
    def _server(self):
        while True:
            # peer_sock, peer_name = await self.loop.sock_accept(self._serv_sock)
            peer_sock, peer_name = yield from self.loop.sock_accept(self._serv_sock)

            peer_sock.setblocking(0)
            peer = Peer(self, peer_sock, peer_name)
            self._peers.append(peer)

            message = 'Peer %s connected!\n' % (peer.name,)
            print(message)

            self.broadcast(message)

    async def input_peers_info(self):
        # Listen for user input and process it
        while True:
            print("Enter 'check' to see connected peers: ")
            user_input = await self.loop.run_in_executor(None, input)
            if user_input == "check":
                print("---------------------------------")
                print("Peers in the network: ")
                print("Number of Peers: ", len(self._peers))
                for i in range(len(self._peers)):
                    print(self._peers[i].name)
                print("---------------------------------")
            elif user_input == "task":
                self._taskManager.check_task()
