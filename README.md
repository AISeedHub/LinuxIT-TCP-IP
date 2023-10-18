# Socket with asyncio

Lý do sử dụng asyncio with non-blocking socket `socket.setblocking(flag=False)` instead of threading:
- Threading: mỗi kết nối sẽ tạo ra 1 thread mới, nếu có nhiều kết nối thì sẽ tạo ra nhiều thread, dẫn đến tốn tài nguyên
- Asyncio: sử dụng 1 thread duy nhất, nó sẽ chạy từng task một, khi task nào đang chờ thì nó sẽ chuyển sang task khác, khi task đó hoàn thành thì nó sẽ quay lại task đang chờ trước đó
- Asyncio sử dụng non-blocking socket, nghĩa là khi có nhiều kết nối đến, nó sẽ chuyển đổi giữa các kết nối đó, không cần phải chờ kết nối trước đó hoàn thành
- Và trên server các tác vụ chỉ là điều kiển một đối tượng duy nhất là Deep Learning Model, và các kết nối đến từ Clients để giám sát nó (nghĩa là không phát sinh thêm các hành động độc lập khác ở mỗi kết nối mới được thiết lập) nên không cần phải sử dụng nhiều thread. 

## Response Error
- `CodeError`: 41(0x29) - `Error`: 0x29 - `Message`: `Invalid request code`(Command is not understanding) or Many request with same code
- `CodeError`: 42(0x2A) - `Error`: 0x2A - `Message`: `Invalid JSON format``
---
### Citations
[GITHUB](https://github.com/DevStarSJ/Study/blob/master/Blog/Python/Socket/02.chat.md)