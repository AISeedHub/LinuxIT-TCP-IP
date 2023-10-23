import selectors
import socket


class Peer(object):
    def __init__(self, server, sock, name):
        self._taskManager = server._taskManager
        self.name = name
        self._sock = sock
        self._server = server
        self.selector = server.selector

    def read_data(self, connection, mask):
        client_address = connection.getpeername()
        print('read({})'.format(client_address))
        data = connection.recv(self._server._config["BUFFER_SIZE"])
        if data:
            print('  received {!r}'.format(data))
            if data.strip().lower() == b'close':
                print('  closing')
                self.selector.unregister(connection)
                connection.close()
                self._server._peers.remove(self)
            else:
                connection.sendall(data)
        else:
            print('closing')
            self.selector.unregister(connection)
            connection.close()

    def broadcast(self, data):
        return self._sock.sendall(data.encode(self._server._config["ENCODING"]))
        # return self.loop.sock_sendall(self._sock, data.encode(self._server._config["ENCODING"]))

    # @coroutine
    def peer_handler(self, connection, mask):
        try:
            print("Peer handler")
            self._peer_loop()
        except IOError as e:
            print("IOError: ", e)
            # finally:
            self._server.remove(self)

    # @coroutine
    def _peer_loop(self, ):
        # while True:
        # buf = yield from self.loop.sock_recv(self._sock, self._server._config["BUFFER_SIZE"])
        data = self._sock.recv(self._server._config["BUFFER_SIZE"])
        # buf = await self.loop.sock_recv(self._sock, self._server._config["BUFFER_SIZE"])
        if data:
            data = data.decode(self._server._config["ENCODING"])
            message = '<----------- %s: %s' % (self.name, data)

            if data.lower() == 'close':
                print('  closing')
                self.selector.unregister(self._sock)
                self._sock.close()
                self._server._peers.remove(self)
            else:
                self.broadcast(data)

            print(message)
            json_response = self._taskManager.distribute_task(data)
            print("json_response: ", json_response)
            # Return the message to the client
            self._server.broadcast(json_response)
        else:
            print("Connection closed by client")
            self._server.remove(self)
            # break


class Server(object):
    def __init__(self, manager):
        self.selector = selectors.DefaultSelector()
        self._taskManager = manager
        # self.loop = manager.loop  # Event loops schedules and manages tasks
        self._config = manager.config
        # Set up the server
        self._serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._serv_sock.setblocking(False)
        # self._serv_sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self._serv_sock.bind((self._config["IP"], self._config["PORT"]))
        self._serv_sock.listen(self._config["MAX_CONNECTIONS"])
        # Set up the selectors "bag of sockets"
        self._peers = []

    def accept_connection(self, sock, mask):
        new_connection, addr = sock.accept()
        print('WELLCOME bro: ({})'.format(addr))
        peer = Peer(self, new_connection, addr)
        new_connection.setblocking(False)
        self._peers.append(peer)
        self.selector.register(new_connection, selectors.EVENT_READ, peer.peer_handler)
        # self.selector.register(new_connection, selectors.EVENT_READ, self.read_data)

    # def read_data(self, connection, mask):
    #     client_address = connection.getpeername()
    #     print("List all connection: ", len(self._peers))
    #     for i in range(len(self._peers)):
    #         print(self._peers[i])
    #     print('read({})'.format(client_address))
    #     data = connection.recv(1024)
    #     if data:
    #         print('  received {!r}'.format(data))
    #         if data.strip().lower() == b'close':
    #             print('  closing')
    #             self.selector.unregister(connection)
    #             connection.close()
    #             self._peers.remove(connection)
    #         else:
    #             connection.sendall(data)
    #     else:
    #         print('closing')
    #         self.selector.unregister(connection)
    #         connection.close()

    def start(self):
        print("Server Address:", self._config["IP"] + ":" + str(self._config["PORT"]))
        print("Start the server...")
        self.selector.register(self._serv_sock, selectors.EVENT_READ, self.accept_connection)
        try:
            while True:
                print("Waiting for connection...")
                for key, mask in self.selector.select():
                    callback = key.data
                    callback(key.fileobj, mask)
        except KeyboardInterrupt:
            print('shutting down')
            self.selector.close()

        # return Task(self._server(), name="server")

    def start_input(self):
        print("Start input listener...")
        # return Task(self.input_peers_info(), name="input_peers_info")

    def remove(self, peer):
        self.selector.unregister(peer._sock)
        peer._sock.close()
        self._peers.remove(peer)
        # print("Removing peer... ", peer.name)
        # print("Bye bye %s" % peer.name)
        # self.broadcast('Peer %s quit!\n' % peer.name)
        # self.peer._sock.close()
        # print(f"Connection with {self.name} closed.")
        # self._peers.remove(peer)

    def broadcast(self, message):
        print("Broadcasting message...")
        for peer in self._peers:
            print(f'----------> {peer.name} : "{message}"')
            peer._sock.sendall(message.encode(self._config["ENCODING"]))
            # peer.send(message)
            print("Done sending")

    # @coroutine
    def _server(self):
        while True:
            # peer_sock, peer_name = await self.loop.sock_accept(self._serv_sock)
            peer_sock, peer_name = yield from self.loop.sock_accept(self._serv_sock)

            peer_sock.setblocking(0)
            peer = Peer(self, peer_sock, peer_name)
            self._peers.append(peer)

            # message = 'WELLCOME Peer %s connected!\n' % (peer.name,)
            # print(message)

            # self.broadcast(message)
            self.remove(peer)
            print("Pass remove")

    # async def input_peers_info(self):
    #     # Listen for user input and process it
    #     while True:
    #         print("Enter 'check' to see connected peers: ")
    #         user_input = await self.loop.run_in_executor(None, input)
    #         if user_input == "check":
    #             print("---------------------------------")
    #             print("Peers in the network: ")
    #             print("Number of Peers: ", len(self._peers))
    #             for i in range(len(self._peers)):
    #                 print(self._peers[i].name)
    #             print("---------------------------------")
    #         elif user_input == "task":
    #             self._taskManager.check_task()