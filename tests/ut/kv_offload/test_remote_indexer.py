import socket
import threading

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.remote_indexer import (
    MAX_MESSAGE_BYTES,
    RemoteIndexerClient,
    _run_graph_callback,
    recv_framed,
    send_framed,
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
        received.append(recv_framed(server_sock))
        send_framed(server_sock, {"ok": True, "op": "reset_cache"})

    thread = threading.Thread(target=serve_reset)
    thread.start()
    client.reset_cache()
    thread.join()
    client_sock.close()
    server_sock.close()

    assert received == [
        {"op": "reset_cache", "version": 1, "rank": 5},
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
            "version": 1,
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
