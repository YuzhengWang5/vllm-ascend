import socket
import threading

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.remote_indexer import (
    CONTROL_MESSAGE,
    MAX_MESSAGE_BYTES,
    RemoteIndexerClient,
    SELECT_CONTEXT_TENSORS,
    SELECT_MESSAGE,
    _recv_exact,
    _run_graph_callback,
    recv_framed,
    recv_raw_select_request,
    recv_raw_select_response,
    send_framed,
    send_raw_select_request,
    send_raw_select_response,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.remote_indexer_service import (
    _resolve_select_context,
)


def test_framed_torch_payload_round_trip():
    sender, receiver = socket.socketpair()
    payload = {
        "op": "select",
        "rank": 3,
        "query": torch.arange(12, dtype=torch.int8).view(3, 4),
    }
    thread = threading.Thread(target=send_framed, args=(sender, payload))
    thread.start()
    result = recv_framed(receiver)
    thread.join()
    sender.close()
    receiver.close()

    assert result["op"] == "select"
    assert result["rank"] == 3
    torch.testing.assert_close(result["query"], payload["query"])


def test_raw_select_request_and_response_round_trip():
    sender, receiver = socket.socketpair()
    tensors = {
        "q": torch.arange(24, dtype=torch.int8).view(2, 3, 4),
        "q_scale": torch.arange(6, dtype=torch.float16).view(2, 3),
        "weights": torch.arange(6, dtype=torch.float16).view(2, 3) + 1,
        "new_k": torch.arange(8, dtype=torch.int8).view(2, 4),
        "new_k_scale": torch.ones((2, 1), dtype=torch.float16),
        "slot_mapping": torch.tensor([7, 9], dtype=torch.int64),
        "actual_seq_lengths_query": torch.tensor([1, 2], dtype=torch.int32),
        "actual_seq_lengths_key": torch.tensor([129, 130], dtype=torch.int32),
        "block_table": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    }
    thread = threading.Thread(
        target=send_raw_select_request,
        args=(sender, 17, 4, tensors),
    )
    thread.start()
    assert _recv_exact(receiver, 1) == SELECT_MESSAGE
    request = recv_raw_select_request(
        receiver,
        lambda _name, shape, dtype: torch.empty(shape, dtype=dtype),
    )
    thread.join()

    assert request["request_id"] == 17
    assert request["layer_id"] == 4
    assert request["include_context"] is True
    for name, expected in tensors.items():
        torch.testing.assert_close(request[name], expected)

    thread = threading.Thread(
        target=send_raw_select_request,
        args=(sender, 18, 5, tensors),
        kwargs={"include_context": False},
    )
    thread.start()
    assert _recv_exact(receiver, 1) == SELECT_MESSAGE
    reuse_request = recv_raw_select_request(
        receiver,
        lambda _name, shape, dtype: torch.empty(shape, dtype=dtype),
    )
    thread.join()
    assert reuse_request["request_id"] == 18
    assert reuse_request["layer_id"] == 5
    assert reuse_request["include_context"] is False
    for name in SELECT_CONTEXT_TENSORS:
        assert name not in reuse_request

    topk = torch.arange(10, dtype=torch.int32).view(2, 1, 5)
    output = torch.empty_like(topk)
    thread = threading.Thread(
        target=send_raw_select_response,
        args=(receiver, 17, topk),
    )
    thread.start()
    recv_raw_select_response(sender, 17, output)
    thread.join()
    torch.testing.assert_close(output, topk)
    sender.close()
    receiver.close()


def test_resolve_select_context_requires_initialization_and_reuses_values():
    with pytest.raises(RuntimeError, match="before initialization"):
        _resolve_select_context({"include_context": False, "layer_id": 1}, None)

    context = {
        "slot_mapping": torch.tensor([7, 9]),
        "actual_seq_lengths_query": torch.tensor([1, 2], dtype=torch.int32),
        "actual_seq_lengths_key": torch.tensor([129, 130], dtype=torch.int32),
        "block_table": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    }
    full_request = {"include_context": True, "layer_id": 0, **context}
    resolved, cached = _resolve_select_context(full_request, None)
    assert resolved is full_request
    for name in SELECT_CONTEXT_TENSORS:
        assert cached[name] is context[name]

    reuse_request = {"include_context": False, "layer_id": 1}
    resolved, reused_cache = _resolve_select_context(reuse_request, cached)
    assert reused_cache is cached
    for name in SELECT_CONTEXT_TENSORS:
        assert resolved[name] is context[name]


def test_graph_callback_dispatches_to_rank_local_client(monkeypatch):
    client = RemoteIndexerClient(
        host="127.0.0.1",
        port=26000,
        rank=7,
        topk=2048,
        connect_timeout_s=1.0,
    )
    called: list[tuple[int, bool]] = []

    def record_callback(layer_id: int) -> None:
        called.append((layer_id, torch.is_inference_mode_enabled()))

    monkeypatch.setattr(client, "_select_graph_host", record_callback)

    _run_graph_callback(7, 23)

    assert called == [(23, True)]


def test_stage_converts_indexer_weights_to_eager_op_dtype():
    client = RemoteIndexerClient(
        host="127.0.0.1",
        port=26000,
        rank=8,
        topk=2048,
        connect_timeout_s=1.0,
    )
    weights = torch.tensor([[1.25, -0.75]], dtype=torch.bfloat16)

    staged = client._stage("weights", weights.to(torch.float16), capturing=False)

    assert staged.dtype == torch.float16
    torch.testing.assert_close(staged, weights.to(torch.float16))


def test_reset_cache_uses_rank_local_connection():
    client_sock, server_sock = socket.socketpair()
    client = RemoteIndexerClient(
        host="127.0.0.1",
        port=26000,
        rank=5,
        topk=2048,
        connect_timeout_s=1.0,
    )
    client._socket = client_sock
    received = []

    def serve_reset() -> None:
        assert _recv_exact(server_sock, 1) == CONTROL_MESSAGE
        received.append(recv_framed(server_sock))
        send_framed(server_sock, {"ok": True, "op": "reset_cache"})

    thread = threading.Thread(target=serve_reset)
    thread.start()
    client.reset_cache()
    thread.join()
    client_sock.close()
    server_sock.close()

    assert received == [
        {"op": "reset_cache", "version": 3, "rank": 5},
    ]


def test_fill_blocks_uses_rank_local_connection_and_deduplicates():
    client_sock, server_sock = socket.socketpair()
    client = RemoteIndexerClient(
        host="127.0.0.1",
        port=26000,
        rank=6,
        topk=2048,
        connect_timeout_s=1.0,
    )
    client._socket = client_sock
    received = []

    def serve_fill() -> None:
        assert _recv_exact(server_sock, 1) == CONTROL_MESSAGE
        received.append(recv_framed(server_sock))
        send_framed(server_sock, {"ok": True, "op": "fill_blocks"})

    thread = threading.Thread(target=serve_fill)
    thread.start()
    client.fill_blocks([7, 3, 7])
    thread.join()
    client_sock.close()
    server_sock.close()

    assert received == [
        {
            "op": "fill_blocks",
            "version": 3,
            "rank": 6,
            "block_ids": [3, 7],
        },
    ]


def test_framed_payload_rejects_oversized_header():
    sender, receiver = socket.socketpair()
    sender.sendall((MAX_MESSAGE_BYTES + 1).to_bytes(8, "big"))
    with pytest.raises(ValueError, match="too large"):
        recv_framed(receiver)
    sender.close()
    receiver.close()
