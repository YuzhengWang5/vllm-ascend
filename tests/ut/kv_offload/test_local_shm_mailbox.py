import ctypes
import threading

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.local_shm_mailbox import (
    LocalShmMailboxClient,
    LocalShmMailboxServer,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.memfabric_mailbox import (
    _REQUEST_HEADER,
    MAILBOX_POOL_BYTES,
    PAYLOAD_OFFSET,
    MemfabricMailboxServer,
    _copy_tensor_bytes,
    _request_specs,
    request_header,
)


class _Buffers:
    def __init__(self) -> None:
        self.values = {}

    def get(self, name, shape, dtype):
        key = (name, shape, dtype)
        if key not in self.values:
            self.values[key] = torch.empty(shape, dtype=dtype)
        return self.values[key]


def _request():
    return {
        "q": torch.arange(32, dtype=torch.int8).reshape(2, 2, 8),
        "q_scale": torch.ones((2, 2), dtype=torch.float16),
        "weights": torch.full((2, 2), 2, dtype=torch.float16),
        "new_k": torch.arange(16, dtype=torch.int8).reshape(2, 8),
        "new_k_scale": torch.ones((2, 1), dtype=torch.float16),
        "slot_mapping": torch.tensor([1, 3], dtype=torch.int64),
        "actual_seq_lengths_query": torch.tensor([1, 2], dtype=torch.int32),
        "actual_seq_lengths_key": torch.tensor([7, 8], dtype=torch.int32),
        "block_table": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    }


def test_local_shm_mailbox_round_trip(tmp_path):
    path = str(tmp_path / "indexer.mailbox")
    server = LocalShmMailboxServer(path, timeout_s=2.0)
    client = LocalShmMailboxClient(path, timeout_s=2.0)
    buffers = _Buffers()
    expected = torch.arange(16, dtype=torch.int32).reshape(2, 1, 8)
    observed_request = {}

    def serve_once():
        while not observed_request:
            request = server.try_receive(buffers.get)
            if request is not None:
                observed_request.update(request)
        server.respond(observed_request["request_id"], expected)

    thread = threading.Thread(target=serve_once)
    thread.start()
    output = torch.empty_like(expected)
    request = _request()
    client.select(1, 4, request, output)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert observed_request["request_id"] == 1
    assert observed_request["layer_id"] == 4
    for name, value in request.items():
        assert torch.equal(observed_request[name], value)
    assert torch.equal(output, expected)

    client.close()
    server.close(unlink=True)


def test_local_shm_mailbox_packed_round_trip(tmp_path):
    path = str(tmp_path / "packed-indexer.mailbox")
    server = LocalShmMailboxServer(path, timeout_s=2.0)
    client = LocalShmMailboxClient(path, timeout_s=2.0)
    expected = torch.arange(16, dtype=torch.int32).reshape(2, 1, 8)
    observed = {}

    def serve_once():
        while not observed:
            request = server.try_receive_packed()
            if request is not None:
                observed.update(request)
        ctypes.memmove(
            server.response_payload_address,
            expected.data_ptr(),
            expected.numel() * expected.element_size(),
        )
        server.publish_response(observed["request_id"])

    thread = threading.Thread(target=serve_once)
    thread.start()
    output = torch.empty_like(expected)
    request = _request()
    client.select(3, 7, request, output)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert observed["request_id"] == 3
    assert observed["layer_id"] == 7
    assert observed["tokens"] == 2
    header_values = _REQUEST_HEADER.unpack(observed["header"])
    expected_bytes, _ = _request_specs(header_values[2:])
    assert observed["payload_bytes"] == expected_bytes
    assert torch.equal(output, expected)

    client.close()
    server.close(unlink=True)


def test_memfabric_server_returns_mapped_tensor_views():
    backing = (ctypes.c_ubyte * MAILBOX_POOL_BYTES)()
    server = MemfabricMailboxServer.__new__(MemfabricMailboxServer)
    server.local_va = ctypes.addressof(backing)
    server._last_sequence = 0
    server._mapped_views = {}
    request = _request()
    header = request_header(11, 9, request)
    ctypes.memmove(server.local_va, header, len(header))
    values = _REQUEST_HEADER.unpack(header)
    total, specs = _request_specs(values[2:])
    offset = 0
    for name, _shape, _dtype, size in specs:
        _copy_tensor_bytes(
            server.local_va + PAYLOAD_OFFSET + offset,
            request[name],
            size,
        )
        offset += size
    assert offset == total

    observed = server.try_receive_views()

    assert observed is not None
    assert observed["request_id"] == 11
    assert observed["layer_id"] == 9
    for name, expected in request.items():
        assert torch.equal(observed[name], expected)
    assert observed["q"].data_ptr() == server.local_va + PAYLOAD_OFFSET

    request["q"][0, 0, 0] = -7
    ctypes.memmove(
        server.local_va + PAYLOAD_OFFSET,
        request["q"].data_ptr(),
        request["q"].numel(),
    )
    assert observed["q"][0, 0, 0].item() == -7


def test_memfabric_response_writes_directly_from_output():
    server = MemfabricMailboxServer.__new__(MemfabricMailboxServer)
    server.peer_gva = 4096
    writes = []
    publications = []
    server._remote_write = lambda source, destination, size: writes.append((source, destination, size))
    server._publish_response = lambda sequence, ok: publications.append((sequence, ok))
    output = torch.arange(16, dtype=torch.int32).reshape(2, 1, 8)

    server.respond(13, output)

    assert writes == [
        (
            output.data_ptr(),
            server.peer_gva + PAYLOAD_OFFSET,
            output.numel() * output.element_size(),
        )
    ]
    assert publications == [(13, True)]
