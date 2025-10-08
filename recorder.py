import time, json, threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Iterable, Optional
import psutil          
import torch.distributed as dist
from schedule_runtime import _mark_done, _mark_done_chunk

@dataclass
class TraceEvent:
    batch_id:  int
    rank:       int
    action:     str
    stage_idx:  int
    mb_idx:     int
    start_ns:   int
    end_ns:     int
    net_series: List[Tuple[int, float, float]] = field(default_factory=list)
    
    # 新增：分块编号（一级命令无分块时为 None）
    chunk:      Optional[int] = None
    # 新增：执行状态（completed / error:...）
    status:     str = "completed"
    # (ts_ns, up_mbps, down_mbps)
    

@dataclass
class SysSample:
    rank: int
    batch_id: int
    time_ns: int
    cpu_utilization: float       # 单位：%
    memory_utilization: float    # 单位：%


class Recorder:
    _NET_ACTIONS = {"SEND_F", "RECV_F", "SEND_B", "RECV_B", "ALL_REDUCE"}

    def __init__(
        self,
        rank: int,
        net_sample_interval_ms: int = 10,
        net_actions: Optional[Iterable[str]] = None,
        measure_net: bool = True,
        mark_actions: bool = True
    ):
        self.rank = rank
        self.events: List[TraceEvent] = []
        self.enabled = True
        self.measure_net = measure_net
        self.sample_interval = max(net_sample_interval_ms, 1) / 1000.0   # 秒
        self.net_actions = set(net_actions) if net_actions else self._NET_ACTIONS
        
        self._sysmon_interval_s = 0.1
        self._sysmon_lock = threading.Lock() # 轮询间隔100ms, 防止出现占用过高的情况
        self._sysmon_active = {}       # batch_id -> threading.Event (Stop sign)
        self._sysmon_threads = {}      # batch_id -> Thread
        self._sysmon_buffer = []       # List[SysSample]
        self.mark_actions = mark_actions


    def set_enabled(self, flag: bool):
        self.enabled = flag

    @contextmanager
    def record(self, batch_id: int, action_id: int,action: str, stage_idx: int, mb_idx: int):
        if not self.enabled:
            yield
            return
        
        self._sysmon_start_if_new(batch_id)

        need_net = self.measure_net and action in self.net_actions
        samples: List[Tuple[int, float, float]] = []
        stop_evt = threading.Event()

        if need_net:
            # 初始累计值
            prev_ts = time.time_ns()
            prev_io = psutil.net_io_counters()
            prev_sent, prev_recv = prev_io.bytes_sent, prev_io.bytes_recv

            def sampler():
                nonlocal prev_ts, prev_sent, prev_recv
                while not stop_evt.is_set():
                    time.sleep(self.sample_interval)
                    curr_ts = time.time_ns()
                    io = psutil.net_io_counters()
                    dt = (curr_ts - prev_ts) / 1e9 or 1e-9
                    up_mbps   = (io.bytes_sent - prev_sent) * 8 / dt / 1e6
                    down_mbps = (io.bytes_recv - prev_recv) * 8 / dt / 1e6
                    samples.append((curr_ts, up_mbps, down_mbps))
                    prev_ts, prev_sent, prev_recv = curr_ts, io.bytes_sent, io.bytes_recv

            thr = threading.Thread(target=sampler, daemon=True)
            thr.start()

        start_ns = time.time_ns()
        try:
            yield
        finally:
            end_ns = time.time_ns()
            if need_net:
                stop_evt.set()
                thr.join()

            # print(f"{action} {mb_idx}计时结束，{action_id}添加表")
            if self.mark_actions:
                _mark_done(batch_id=batch_id,action_id=action_id)
            
            self.events.append(
                TraceEvent(
                    batch_id, self.rank, action, stage_idx, mb_idx,
                    start_ns, end_ns,
                    chunk=None,
                    status="completed",
                    net_series=samples,
                )
            )

    ...
    def record_async(
        self,
        batch_id: int,
        action_id: int, 
        action: str,
        stage_idx: int,
        mb_idx: int,
        works: List[dist.Work],
        start_ns: int,
        poll_interval: float = 0.001,
        chunk_idx: Optional[int] = None,
    ):
        if not self.enabled:
            return
        
        self._sysmon_start_if_new(batch_id)
        
        need_net = self.measure_net and action in self.net_actions
        need_net = False
        samples, stop_evt = [], threading.Event()

        if action in ("SEND_F", "SEND_B"):

            self.events.append(
                TraceEvent(
                    batch_id, self.rank, action, stage_idx, mb_idx,
                    start_ns, start_ns,  # Use start time as end time for now
                    chunk=chunk_idx,
                    status="posted",  
                    net_series=[],
                )
            )
            return  # Exit early
        
        if need_net:
            def sampler():
                prev_ts = time.time_ns()
                io = psutil.net_io_counters()
                prev_sent, prev_recv = io.bytes_sent, io.bytes_recv
                while not stop_evt.is_set():
                    time.sleep(self.sample_interval)
                    curr_ts = time.time_ns()
                    io = psutil.net_io_counters()
                    dt = (curr_ts - prev_ts) / 1e9 or 1e-9
                    up   = (io.bytes_sent - prev_sent) * 8 / dt / 1e6
                    down = (io.bytes_recv - prev_recv) * 8 / dt / 1e6
                    samples.append((curr_ts, up, down))
                    prev_ts, prev_sent, prev_recv = curr_ts, io.bytes_sent, io.bytes_recv
            threading.Thread(target=sampler, daemon=True).start()

        def waiter():
            status = "completed"
            try:
                for w in works:
                    if w.is_completed():
                        continue
                    w.wait()
                end_ns = time.time_ns()
                
                if need_net:
                    stop_evt.set()
                if self.mark_actions:
                    # Mark done after waiting completes
                    if chunk_idx is None:
                        _mark_done(batch_id=batch_id, action_id=action_id)
                    else:
                        # print(f"[{dist.get_rank()}] Marking done chunk for batch={batch_id}, action={action_id}, chunk={chunk_idx}")
                        _mark_done_chunk(batch_id=batch_id, action_id=action_id, chunk_idx=chunk_idx)
                        # print(f"[{self.rank}] DONE {action} st={stage_idx} mb={mb_idx} chunk={chunk_idx} "
                        #     f"works={len(works)}")
            except Exception as e:
                status = f"error:{type(e).__name__}"
                end_ns = time.time_ns()
            finally:
                self.events.append(
                    TraceEvent(
                        batch_id, self.rank, action, stage_idx, mb_idx,
                        start_ns, end_ns,
                        chunk=chunk_idx,
                        status=status,
                        net_series=samples,
                    )
                )
        
        threading.Thread(target=waiter, daemon=True).start()
        
    
        # === 新增：系统资源监控辅助函数 ===
    def _sysmon_start_if_new(self, batch_id: int):
        """若首次见到该 batch_id，则启动一个采样线程；仅追加，不影响旧逻辑。"""
        if not self.enabled:
            return
        with self._sysmon_lock:
            if batch_id in self._sysmon_active:
                return
            stop_evt = threading.Event()
            self._sysmon_active[batch_id] = stop_evt

            def _sampler():
                # 首次调用做一次“预热”，避免第一次 cpu_percent 异常值
                try:
                    psutil.cpu_percent(interval=None)
                except Exception:
                    pass
                while not stop_evt.is_set():
                    t_ns = time.time_ns()
                    try:
                        cpu = float(psutil.cpu_percent(interval=None))  # %
                        mem = float(psutil.virtual_memory().percent)    # %
                    except Exception:
                        cpu, mem = -1.0, -1.0
                    with self._sysmon_lock:
                        self._sysmon_buffer.append(
                            SysSample(self.rank, batch_id, t_ns, cpu, mem)
                        )
                    time.sleep(self._sysmon_interval_s)

            thr = threading.Thread(target=_sampler, daemon=True)
            self._sysmon_threads[batch_id] = thr
            thr.start()

    def _sysmon_stop_all(self):
        """停止所有 batch 的采样线程（用于 dump 切片边界），仅新增不影响旧逻辑。"""
        with self._sysmon_lock:
            for bid, evt in list(self._sysmon_active.items()):
                evt.set()
            for bid, thr in list(self._sysmon_threads.items()):
                try:
                    thr.join(timeout=1.0)
                except Exception:
                    pass
            self._sysmon_active.clear()
            self._sysmon_threads.clear()

    def _sysmon_flush_to_file(self, fname: Optional[str] = None):
        """将缓冲写入独立 JSONL 文件，每行一条 SysSample。"""
        with self._sysmon_lock:
            buf = self._sysmon_buffer
            self._sysmon_buffer = []

        if not buf:
            return

        path = fname or f"cpu_mem_rank{self.rank}.jsonl"
        try:
            with open(path, "a") as f:
                for s in buf:
                    # 序列化为 5 字段的扁平对象
                    rec = {
                        "rank": s.rank,
                        "batch_id": s.batch_id,
                        "time_ns": s.time_ns,
                        "cpu_utilization": f"{float(s.cpu_utilization):.7f}",
                        "memory_utilization": f"{float(s.memory_utilization):.7f}",
                    }
                    f.write(json.dumps(rec) + "\n")
        except Exception:
            # 静默失败，不影响旧逻辑
            pass
    
    def dump(self, fname: str = None):
        fname = fname or f"timeline_rank{self.rank}.json"
        with open(fname, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
        
        self._sysmon_flush_to_file()
        self._sysmon_stop_all()
