"""Process-isolated MemFabric relay for the remote SFA indexer.

The SFA offload allocator and BigMemory both keep process-global MemFabric
state.  A decoder worker therefore cannot safely create the two independent
groups (TP-local offload and rank-paired indexer mailbox) in one process with
MemFabric 1.2.  This relay owns the rank-paired mailbox in a small sidecar
process and exposes the existing binary protocol on localhost to the decoder.
"""

from __future__ import annotations

import argparse
import json
import select as io_select
import socket
import time
import traceback

import torch

from .local_shm_mailbox import LocalShmMailboxServer, default_shm_path
from .memfabric_mailbox import MemfabricMailboxClient
from .remote_indexer import (
    CONTROL_MESSAGE,
    PROTOCOL_VERSION,
    SELECT_MESSAGE,
    _recv_exact,
    recv_framed,
    recv_raw_select_request,
    send_framed,
    send_raw_select_error,
    send_raw_select_response,
)


class _HostBuffers:
    def __init__(self) -> None:
        self._buffers: dict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor] = {}

    def get(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        key = (name, shape, dtype)
        result = self._buffers.get(key)
        if result is None:
            result = torch.empty(shape, dtype=dtype, device="cpu")
            self._buffers[key] = result
        return result


def _connect_mailbox(
    *,
    remote_host: str,
    remote_port: int,
    rank: int,
    device: int,
    store_url: str,
    timeout_s: float,
) -> tuple[socket.socket, MemfabricMailboxClient]:
    connection = socket.create_connection((remote_host, remote_port), timeout_s)
    connection.settimeout(None)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_framed(
        connection,
        {
            "op": "hello",
            "version": PROTOCOL_VERSION,
            "rank": rank,
            "select_transport": "memfabric_mailbox",
            "memfabric_store_url": store_url,
        },
    )
    response = recv_framed(connection)
    expected_response = {
        "ok": True,
        "version": PROTOCOL_VERSION,
        "select_transport": "memfabric_mailbox",
    }
    if response != expected_response:
        raise RuntimeError(f"Remote indexer handshake mismatch: {response!r}")
    mailbox = MemfabricMailboxClient(
        logical_rank=rank,
        store_url=store_url,
        device=device,
        timeout_s=timeout_s,
        defer_create=True,
    )
    send_framed(
        connection,
        {
            "op": "memfabric_rank0_initialized",
            "version": PROTOCOL_VERSION,
            "rank": rank,
        },
    )
    response = recv_framed(connection)
    if response != {"ok": True, "memfabric_rank0_join": True}:
        raise RuntimeError(f"Remote indexer pre-join failed: {response!r}")
    mailbox.create()
    send_framed(
        connection,
        {
            "op": "memfabric_rank0_joined",
            "version": PROTOCOL_VERSION,
            "rank": rank,
        },
    )
    response = recv_framed(connection)
    if response != {"ok": True, "mailbox_ready": True}:
        raise RuntimeError(f"Remote indexer mailbox join failed: {response!r}")
    return connection, mailbox


def _forward_control(inbound: socket.socket, outbound: socket.socket, rank: int) -> None:
    request = recv_framed(inbound)
    if request.get("rank") != rank:
        raise RuntimeError(f"Control rank mismatch: expected={rank}, got={request.get('rank')}")
    outbound.sendall(CONTROL_MESSAGE)
    send_framed(outbound, request)
    send_framed(inbound, recv_framed(outbound))


def _serve_decoder(
    *,
    inbound: socket.socket,
    outbound: socket.socket,
    mailbox: MemfabricMailboxClient,
    rank: int,
    topk: int,
    log_every: int,
    local_mailbox: LocalShmMailboxServer | None,
) -> None:
    hello = recv_framed(inbound)
    expected = {
        "op": "hello",
        "version": PROTOCOL_VERSION,
        "rank": rank,
        "select_transport": "shm_mailbox" if local_mailbox else "raw_tcp",
    }
    if hello != expected:
        raise RuntimeError(f"Decoder handshake mismatch: {hello!r}")
    send_framed(
        inbound,
        {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "select_transport": "shm_mailbox" if local_mailbox else "raw",
        },
    )
    buffers = _HostBuffers()
    while True:
        if local_mailbox is not None:
            readable, _, _ = io_select.select([inbound], [], [], 0)
            if readable:
                kind = _recv_exact(inbound, 1)
            else:
                request = local_mailbox.try_receive_packed()
                if request is None:
                    continue
                request_id = int(request["request_id"])
                try:
                    started_at = time.perf_counter()
                    response_bytes = int(request["tokens"]) * topk * 4
                    mailbox.select_from_address(
                        request_id,
                        request["header"],
                        int(request["payload_address"]),
                        local_mailbox.response_payload_address,
                        response_bytes,
                    )
                    local_mailbox.publish_response(request_id)
                    if request_id % log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "relay_select",
                                    "local_transport": "shm_mailbox",
                                    "rank": rank,
                                    "request_id": request_id - 1,
                                    "layer_id": int(request["layer_id"]),
                                    "tokens": int(request["tokens"]),
                                    "relay_path": "zero_copy",
                                    "elapsed_ms": (time.perf_counter() - started_at) * 1000,
                                }
                            ),
                            flush=True,
                        )
                except Exception:
                    local_mailbox.respond_error(request_id)
                    raise
                continue
        else:
            kind = _recv_exact(inbound, 1)
        if kind == CONTROL_MESSAGE:
            _forward_control(inbound, outbound, rank)
            continue
        if kind != SELECT_MESSAGE:
            raise RuntimeError(f"Unsupported decoder message kind: {kind!r}")
        request_id = 0
        try:
            started_at = time.perf_counter()
            request = recv_raw_select_request(inbound, buffers.get)
            request_id = int(request["request_id"])
            output = buffers.get(
                "topk_response",
                (request["q"].shape[0], 1, topk),
                torch.int32,
            )
            mailbox.select(
                request_id + 1,
                int(request["layer_id"]),
                request,
                output,
            )
            send_raw_select_response(inbound, request_id, output)
            if request_id % log_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "relay_select",
                            "rank": rank,
                            "request_id": request_id,
                            "layer_id": int(request["layer_id"]),
                            "tokens": request["q"].shape[0],
                            "elapsed_ms": (time.perf_counter() - started_at) * 1000,
                        }
                    ),
                    flush=True,
                )
        except Exception as error:
            send_raw_select_error(inbound, request_id, repr(error))
            raise


def serve(args: argparse.Namespace) -> None:
    outbound, mailbox = _connect_mailbox(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        rank=args.rank,
        device=args.device,
        store_url=args.store_url,
        timeout_s=args.connect_timeout,
    )
    local_mailbox = None
    shm_path = args.shm_path or default_shm_path(args.rank)
    if args.local_transport == "shm_mailbox":
        local_mailbox = LocalShmMailboxServer(
            shm_path,
            timeout_s=args.connect_timeout,
        )
    print(
        json.dumps(
            {
                "event": "relay_mailbox_ready",
                "rank": args.rank,
                "remote": f"{args.remote_host}:{args.remote_port}",
                "listen": f"{args.host}:{args.port}",
                "local_transport": args.local_transport,
                "shm_path": shm_path if local_mailbox else None,
            }
        ),
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        inbound, peer = listener.accept()
        with inbound, outbound:
            inbound.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(
                json.dumps({"event": "decoder_connected", "rank": args.rank, "peer": peer[0]}),
                flush=True,
            )
            _serve_decoder(
                inbound=inbound,
                outbound=outbound,
                mailbox=mailbox,
                rank=args.rank,
                topk=args.topk,
                log_every=args.log_every,
                local_mailbox=local_mailbox,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--store-url", required=True)
    parser.add_argument(
        "--local-transport",
        choices=("raw_tcp", "shm_mailbox"),
        default="raw_tcp",
    )
    parser.add_argument("--shm-path", default="")
    parser.add_argument("--topk", default=2048, type=int)
    parser.add_argument("--connect-timeout", default=120.0, type=float)
    parser.add_argument("--log-every", default=100, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        serve(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "relay_error",
                    "rank": args.rank,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
