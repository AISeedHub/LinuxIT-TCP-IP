# Socket with asyncio

Lý do sử dụng asyncio with non-blocking socket `socket.setblocking(flag=False)` instead of threading:

- Threading: mỗi kết nối sẽ tạo ra 1 thread mới, nếu có nhiều kết nối thì sẽ tạo ra nhiều thread, dẫn đến tốn tài nguyên
- Asyncio: sử dụng 1 thread duy nhất, nó sẽ chạy từng task một, khi task nào đang chờ thì nó sẽ chuyển sang task khác,
  khi task đó hoàn thành thì nó sẽ quay lại task đang chờ trước đó
- Asyncio sử dụng non-blocking socket, nghĩa là khi có nhiều kết nối đến, nó sẽ chuyển đổi giữa các kết nối đó, không
  cần phải chờ kết nối trước đó hoàn thành
- Và trên server các tác vụ chỉ là điều kiển một đối tượng duy nhất là Deep Learning Model, và các kết nối đến từ
  Clients để giám sát nó (nghĩa là không phát sinh thêm các hành động độc lập khác ở mỗi kết nối mới được thiết lập) nên
  không cần phải sử dụng nhiều thread.

# Socket with `selectors` lib in python
[Python module selectors](https://mathspp.com/blog/til/022)

```python
# server.py
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 7342))
server.listen()
print("About to accept:")
server.accept()
print("Call to `accept` returned.")
```

If you run this with python server.py, the message "About to accept:" gets printed, but not the message "Call
to `accept` returned.". Why? Because there is no connection to be accept, and so the call to accept blocks, waiting for
a connection.
Thus, we can't just call the method recv on the different clients we have, because if one doesn't have anything for us,
we will block! A common way to deal with this is by spawning a thread for each client, and do the blocking calls in
those separate threads.

Another way to deal with this is with the module `selectors`.
```python

# server.py

import selectors
import socket

# Set up the server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 7342))
server.listen()

# Set up the selectors "bag of sockets"
selector = selectors.DefaultSelector()
selector.register(server, selectors.EVENT_READ)

while True:
    events = selector.select()
    for key, _ in events:
        sock = key.fileobj
        print("About to accept.")
        client, _ = sock.accept()
        print("Accepted.")
```

## Response Error

- `CodeError`: 41(0x29) - `Error`: 0x29 - `Message`: `Invalid request code`(Command is not understanding) or Many
  request with same code
- `CodeError`: 42(0x2A) - `Error`: 0x2A - `Message`: `Invalid JSON format``

---

### Citations

[GITHUB](https://github.com/DevStarSJ/Study/blob/master/Blog/Python/Socket/02.chat.md)