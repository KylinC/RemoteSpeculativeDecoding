import asyncio
import torch
import pickle
import transformers
import functools
import config
import logging
import websockets
import time

from transformers.models.opt import OPTModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer

from websockets import WebSocketClientProtocol as WSClient, WebSocketServerProtocol as WSServer, connect, serve
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Callable, List, Tuple
from dataclasses import dataclass
from remote.utils import timer, get_timer_stats, update_timer


uuid_counter = 0
def uuid():
    global uuid_counter
    uuid_counter += 1
    return uuid_counter


@dataclass
class AsyncContext:
    ids: torch.Tensor
    max_len: int
    gamma: int
    time_stamp: int = 0
    len_verified: int = 0
    len_drafted: int = 0
    # signal: Optional[asyncio.Future] = None


@dataclass
class WsMsg:
    flag: bool                      # require drawback or not
    ids: Optional[torch.Tensor]
    prefix_len: Optional[int]
    gamma: Optional[int]
    time_stamp: int


def load_ws_msg(data) -> WsMsg:
    assert isinstance(data, bytes)
    msg = pickle.loads(data)
    return msg


async def copy_gpu2cpu(x: torch.Tensor) -> torch.Tensor:
    # TODO: mark with events
    return x.cpu()
    x_cpu = x.to("cpu", non_blocking = True)
    stream = torch.cuda.current_stream(x.device)
    stream.synchronize()
    return x_cpu


async def dump_ws_msg(flag: bool, 
                      time_stamp: int,
                      x: Optional[torch.Tensor],
                      prefix_len: Optional[int] = None,
                      gamma: Optional[int] = None) -> bytes:
    if x is not None and x.device.type != "cpu":
        y = await copy_gpu2cpu(x)
        msg = WsMsg(flag, y, prefix_len, gamma, time_stamp)
    else:
        msg = WsMsg(flag, x, prefix_len, gamma, time_stamp)
    data = pickle.dumps(msg)
    return data


class AsyncStreamer:

    def __init__(self, 
                 cur_len: int, 
                 ws_client: WSClient, 
                 parent: "AsyncClient",
                 loop: asyncio.BaseEventLoop,
                 step: int = 1) -> None:
        self.cur_len = cur_len
        self.ws_client = ws_client
        self.parent = parent
        self.loop = loop
        self.step = step

        self.sequences: Optional[torch.Tensor] = None

    def _add_token(self, token: torch.Tensor):
        if self.sequences is None:
            self.sequences = token.reshape([token.shape[0], 1, token.shape[1]])
        else:
            self.sequences = torch.cat([self.sequences, token[:, None]], dim=-1)

    def put(self, token: torch.Tensor):
        self._add_token(token)
        if self.sequences.shape[1] == self.step:
            self.parent._async_spec_from_server(
                self.client,

            )
            self.sequences = None
            
        
        # self._async_spec_from_server(
        #     client,
        #     x,
        #     cur_len,
        #     delta,
        #     time_stamp = self.contexts[cid].time_stamp
        # )

    def end(self):
        pass


class AsyncWrapper:

    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers)

    async def run(self,
                  func: Callable,
                  args: Tuple,
                  event_loop: Optional[asyncio.BaseEventLoop] = None):
        if event_loop is None:
            event_loop = asyncio.get_event_loop()
        return await event_loop.run_in_executor(self._pool, functools.partial(func, *args))


class AsyncClient:
    
    def __init__(self, 
                 server_url: str, 
                 model: OPTModel) -> None:
        self.model = model
        self.url = server_url
        self.contexts: Dict[int, AsyncContext] = {}
        self.loop = asyncio.get_event_loop()

    def decode(self, 
               prefix: torch.Tensor,
               max_len: int,
               gamma: int = 4) -> torch.Tensor:
        cid = uuid()
        self.contexts[cid] = AsyncContext(
            prefix,
            max_len,
            gamma
        )
        self.loop.run_until_complete(self._async_decode(cid))
        return self.contexts[cid].ids
        
    async def _async_recv(self, 
                          cid: int,
                          client: WSClient):
        while True:
            try:
                msg = load_ws_msg(await client.recv())
                if msg.flag:
                    # wrong prediction, require drawback
                    self.contexts[cid].ids = msg.ids.to(self.model.device, non_blocking=True)
                    self.contexts[cid].len_drafted = msg.ids.shape[1]
                    self.contexts[cid].time_stamp += 1
                    
                    # TODO: check if non_blocking works
                    print("drawback!")
                self.contexts[cid].len_verified = msg.ids.shape[1]
                print("verified: ", self.contexts[cid].len_verified)
            except asyncio.exceptions.CancelledError:
                break
            except:
                logging.exception("Found exception")
                break

    async def _async_decode(self, cid):
        async with connect(self.url) as client:
            coro = asyncio.create_task(self._async_recv(cid, client))

            await self._async_block_decoding(cid, client)
            coro.cancel()

    def _async_spec_from_server(self, 
                                client: WSClient,
                                x: torch.Tensor,
                                prefix_len: int,
                                gamma: int,
                                time_stamp: int):
        async def _spec_internal():
            try:
                await client.send(
                    await dump_ws_msg(
                        flag = False,
                        time_stamp = time_stamp,
                        x = x,
                        prefix_len = prefix_len,
                        gamma = gamma
                    )
                )
            except websockets.exceptions.ConnectionClosedOK:
                pass
        asyncio.create_task(_spec_internal())

    async def _async_block_decoding(self, 
                                    cid: int,
                                    client: WSClient) -> torch.Tensor:
        ctx: AsyncContext = self.contexts[cid]
        max_len = ctx.ids.shape[1] + ctx.max_len
        gamma = ctx.gamma

        self._async_spec_from_server(
            client=client,
            x=ctx.ids,
            prefix_len=ctx.ids.shape[1],
            gamma=ctx.ids.shape[1],
            time_stamp=-1
        )

        print("before:", ctx)
        timer(None)

        st = time.time()

        while self.contexts[cid].len_verified < max_len:
            await asyncio.sleep(0)  # yield to other coroutine, may retreat
            ctx = self.contexts[cid]
            x = self.contexts[cid].ids
            cur_len = x.shape[1]

            if cur_len == max_len:
                # print("reach maximal length, skipped")
                if st != 0:
                    update_timer("draft_done", time.time() - st)
                st = 0
                continue

            print("max_len:", max_len, "current len:", cur_len, "verified len:", ctx.len_verified, "drafted len:", ctx.len_drafted)

            delta = min(gamma, max_len - cur_len)

            timer("idle")
            x = self.model.generate(x, max_length = cur_len + delta)
            timer("generate")

            self._async_spec_from_server(
                client,
                x,
                cur_len,
                delta,
                time_stamp = self.contexts[cid].time_stamp
            )

            self.contexts[cid].ids = x
            self.contexts[cid].len_drafted = x.shape[1]

        print("after:", self.contexts[cid])
        print(get_timer_stats())


class AsyncServer:

    def __init__(self, model_name) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model: OPTModel = AutoModelForCausalLM.from_pretrained(model_name,
                                                                     torch_dtype=torch.float16,
                                                                     trust_remote_code=True).to(self._device)                                                   
        logging.info(f"loaded model: {model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._server_stop_flag: asyncio.Future
        self._sync_wrapper = AsyncWrapper()
        self._kv_cache: Optional[torch.Tensor] = None

    def spec_tokens(self, msg: WsMsg):
        print("spec_tokens len:", msg.prefix_len)
        x, prefix_len, gamma = msg.ids, msg.prefix_len, msg.gamma
        x_pre = x[..., : x.shape[1] - gamma]
        x = x[..., x.shape[1] - gamma:]

        if x.device != self._model.device:
            x = x.to(self._model.device)
        
        # TODO: reuse the kvcache
        with torch.no_grad():
            output: CausalLMOutputWithPast = self._model(x, use_cache=True, past_key_values=self._kv_cache)
            y = output.logits.argmax(dim=2)
            self._kv_cache = output.past_key_values
        # print("current kv_cache shape:", self._kv_cache.shape)

        if msg.time_stamp >= 0:
            n = 0
            flag = False

            for _ in range(gamma):
                if y[0][n] == x[0][n]:
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    print(f"reject {n}")
                    flag = True
                    x[0][n] = y[0][n]
                    break
        else:
            n = x.shape[1]

        ids = torch.concat([x_pre, x[:, :n].to('cpu')], dim=1)
        return flag, ids

    async def _run_server_internal(self):
        self._server_stop_flag = asyncio.get_event_loop().create_future()

        async def server_handler(ws: WSServer):
            ts = -1e100

            banned: Dict[int, bool] = {}
            self._kv_cache = None
            async for data in ws:
                msg = load_ws_msg(data)
                print("receive drafted length:", msg.prefix_len)
                if msg.time_stamp > ts:
                    ts = msg.time_stamp
                elif msg.time_stamp < ts or ts in banned:
                    print("current ts,", ts, "ignore pack with ts", msg.time_stamp)
                    continue

                flag, ids = await self._sync_wrapper.run(
                    func = self.spec_tokens,
                    args = (msg, ),
                    event_loop = asyncio.get_event_loop()
                )
                if flag:
                    banned[ts] = True

                await ws.send(await dump_ws_msg(flag, ts, ids))
                print("sent results:", ts, ids.shape[1])

        async with serve(server_handler, config.WS_SERVER_ADDR, config.WS_SERVER_PORT):
            await self._server_stop_flag

    def run_server(self):
        asyncio.run(self._run_server_internal())
