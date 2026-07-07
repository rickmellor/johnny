"""Host inventory — CPU, memory, storage, network (the non-GPU half of `johnny hinfo`).

Same discipline as hardware/detect.py: host-side, stdlib + light CLIs (lscpu, lsblk,
dmidecode, /proc, /sys), every source degrades to a partial/empty result rather than
raising. Derived metrics are only ever *honest*:
- NIC throughput is the real negotiated link speed (/sys/class/net/*/speed).
- CPU 'MIPS' is BogoMIPS — the conventional Linux proxy, labelled as such (it is not a
  benchmarked IPC figure).
- Memory type/speed/bandwidth come from dmidecode, which needs root; unprivileged we show
  capacity only and say so. Bandwidth is a theoretical peak (channels x MT/s x 8), flagged.
- Disk 'throughput' is a coarse bus-class estimate (NVMe/SATA/HDD), flagged — not measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..util import run, which

# CPU flags that matter for AI inference throughput, in report order (subset of the
# hundreds in /proc/cpuinfo — we only surface the ones that change kernel selection).
_AI_FLAGS = [
    "amx_int8", "amx_bf16", "amx_tile", "avx512_vnni", "avx512_bf16", "avx512f",
    "avx_vnni", "avx2", "fma", "f16c", "sse4_2",
]

# Coarse per-bus sequential-throughput class (labelled an estimate, never measured).
_DISK_CLASS = {"NVMe SSD": "~3–7 GB/s", "SATA SSD": "~550 MB/s", "HDD": "~150 MB/s"}


@dataclass
class CPUInfo:
    model: str
    sockets: int
    cores: int          # physical cores (all sockets)
    threads: int        # logical processors
    base_mhz: float | None
    max_mhz: float | None
    l3_mb: float | None
    bogomips_total: float | None   # per-core BogoMIPS x logical count
    ai_flags: list[str] = field(default_factory=list)


@dataclass
class DIMM:
    size_gb: float
    type: str | None
    speed_mts: int | None            # rated (SPD) speed
    configured_mts: int | None = None  # actual running speed
    part_number: str | None = None
    rank: int | None = None
    locator: str | None = None
    manufacturer: str | None = None


@dataclass
class MemInfo:
    total_gb: float                       # usable, from /proc/meminfo
    dimms: list[DIMM] = field(default_factory=list)
    mem_type: str | None = None
    speed_mts: int | None = None          # fastest configured speed
    populated: int | None = None          # populated DIMM slots
    slots: int | None = None              # total DIMM slots (memory array)
    max_capacity_gb: float | None = None
    ecc: str | None = None
    cached: bool = False                  # rendered from a seeded cache, not a live read
    captured_at: str | None = None
    privileged: bool = True               # False -> no DIMM detail (capacity only)


@dataclass
class Disk:
    name: str
    size_gb: float
    kind: str            # "NVMe SSD" | "SATA SSD" | "HDD" | "SSD"
    model: str
    throughput_est: str | None = None


@dataclass
class NIC:
    name: str
    speed_mbps: int | None   # negotiated link speed; None/-1 when down/unknown
    state: str               # up | down | ...
    mac: str | None


@dataclass
class HostInfo:
    cpu: CPUInfo
    mem: MemInfo
    disks: list[Disk] = field(default_factory=list)
    nics: list[NIC] = field(default_factory=list)


# --------------------------------------------------------------------------- CPU
def _cpuinfo() -> tuple[dict, int, float | None, list[str]]:
    """(first-processor fields, logical count, per-core bogomips, flags) from /proc/cpuinfo."""
    first: dict[str, str] = {}
    logical = 0
    bogo: float | None = None
    flags: list[str] = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" not in line:
                continue
            k, v = (x.strip() for x in line.split(":", 1))
            if k == "processor":
                logical += 1
            elif not first.get(k):
                first[k] = v
            if k == "bogomips" and bogo is None:
                try:
                    bogo = float(v)
                except ValueError:
                    pass
            if k == "flags" and not flags:
                flags = v.split()
    except OSError:
        pass
    return first, logical, bogo, flags


def _lscpu() -> dict:
    rc, out, _ = run(["lscpu"], timeout=10)
    d: dict[str, str] = {}
    if rc == 0:
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def cpu_info() -> CPUInfo:
    first, logical, bogo, flags = _cpuinfo()
    ls = _lscpu()

    def _num(*keys, cast=int, default=None):
        for k in keys:
            if k in ls:
                try:
                    return cast(float(ls[k]))
                except ValueError:
                    pass
        return default

    sockets = _num("Socket(s)", default=1) or 1
    cores_per_socket = _num("Core(s) per socket", default=None)
    cores = (cores_per_socket * sockets) if cores_per_socket else (logical // 2 or logical)
    l3 = None
    if "L3 cache" in ls:  # e.g. "64 MiB (4 instances)"
        try:
            l3 = float(ls["L3 cache"].split()[0])
        except (ValueError, IndexError):
            l3 = None
    return CPUInfo(
        model=ls.get("Model name") or first.get("model name") or "unknown CPU",
        sockets=int(sockets),
        cores=int(cores),
        threads=logical,
        base_mhz=(float(first["cpu MHz"]) if first.get("cpu MHz") else None),
        max_mhz=_num("CPU max MHz", cast=float),
        l3_mb=l3,
        bogomips_total=(round(bogo * logical, 0) if bogo else None),
        ai_flags=[f for f in _AI_FLAGS if f in flags],
    )


# --------------------------------------------------------------------------- memory
def _host_ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    return 0.0


def _dmi_blocks(text: str):
    """Yield (dmi_type, {key: value}) per dmidecode handle block."""
    import re

    dtype, kv = None, {}
    for line in text.splitlines():
        if line.startswith("Handle "):
            if kv:
                yield dtype, kv
            m = re.search(r"DMI type (\d+)", line)
            dtype, kv = (int(m.group(1)) if m else None), {}
        elif "\t" in line and ":" in line:
            k, v = line.strip().split(":", 1)
            kv[k.strip()] = v.strip()
    if kv:
        yield dtype, kv


def _int_or_none(s: str | None) -> int | None:
    try:
        return int(s) if s and s.strip().isdigit() else None
    except (ValueError, AttributeError):
        return None


def parse_dmidecode(text: str, total_gb: float, captured_at: str | None = None) -> MemInfo:
    """Parse `dmidecode -t memory`: the Physical Memory Array (type 16 — slots, max
    capacity, ECC) plus each populated Memory Device (type 17)."""
    info = MemInfo(total_gb=total_gb, captured_at=captured_at)
    dimms: list[DIMM] = []
    for dtype, kv in _dmi_blocks(text):
        if dtype == 16:
            info.max_capacity_gb = _size_to_gb(kv.get("Maximum Capacity", "")) or None
            info.slots = _int_or_none(kv.get("Number Of Devices"))
            info.ecc = kv.get("Error Correction Type")
        elif dtype == 17:
            size = kv.get("Size", "")
            if not size or size.lower() in ("no module installed", "not installed"):
                continue
            dimms.append(DIMM(
                size_gb=_size_to_gb(size),
                type=kv.get("Type"),
                speed_mts=_speed_mts(kv.get("Speed", "")),
                configured_mts=_speed_mts(kv.get("Configured Memory Speed", "")),
                part_number=(kv.get("Part Number") or "").strip() or None,
                rank=_int_or_none(kv.get("Rank")),
                locator=kv.get("Locator"),
                manufacturer=(kv.get("Manufacturer") or "").strip() or None,
            ))
    info.dimms = dimms
    if dimms:
        info.populated = len(dimms)
        confs = [d.configured_mts or d.speed_mts for d in dimms if (d.configured_mts or d.speed_mts)]
        info.speed_mts = max(confs) if confs else None
        info.mem_type = next((d.type for d in dimms if d.type), None)
    return info


def _mem_cache_path(state_dir) -> Path:
    return Path(state_dir) / "hostinfo" / "memory.json"


def _write_mem_cache(state_dir, info: MemInfo) -> None:
    if not state_dir:
        return
    from dataclasses import asdict

    try:
        p = _mem_cache_path(state_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        import json

        p.write_text(json.dumps(asdict(info), indent=2))
    except OSError:
        pass


def _read_mem_cache(state_dir) -> MemInfo | None:
    if not state_dir:
        return None
    import json

    try:
        d = json.loads(_mem_cache_path(state_dir).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    dimms = [DIMM(**x) for x in d.pop("dimms", [])]
    try:
        return MemInfo(dimms=dimms, **d)
    except TypeError:  # cache from an older schema — ignore it
        return None


def _today() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def mem_info(state_dir=None) -> MemInfo:
    """Live DIMM detail when running as root (auto-seeding the cache), else the seeded
    cache from a prior `--seed-memory`, else capacity-only."""
    total = _host_ram_gb()
    if which("dmidecode"):
        rc, out, _ = run(["dmidecode", "-t", "memory"], timeout=10)
        if rc == 0 and out.strip():
            info = parse_dmidecode(out, total, captured_at=_today())
            _write_mem_cache(state_dir, info)  # so future unprivileged runs have it
            return info
    cached = _read_mem_cache(state_dir)
    if cached and cached.dimms:
        cached.total_gb = total  # keep the live usable total; DIMM detail is from the seed
        cached.cached = True
        cached.privileged = True
        return cached
    return MemInfo(total_gb=total, privileged=False)


def seed_memory(state_dir) -> MemInfo | None:
    """Run `dmidecode -t memory` with sudo (prompts for a password), parse it, and cache
    the DIMM inventory so later unprivileged `hinfo` runs can render it. None on failure."""
    if not which("dmidecode"):
        return None
    import os

    cmd = ["dmidecode", "-t", "memory"]
    if os.geteuid() != 0:
        cmd = ["sudo", *cmd]
    rc, out, _ = run(cmd, timeout=120)  # generous: the user may be typing a sudo password
    if rc != 0 or not out.strip():
        return None
    info = parse_dmidecode(out, _host_ram_gb(), captured_at=_today())
    _write_mem_cache(state_dir, info)
    return info


def _size_to_gb(s: str) -> float:
    parts = s.split()
    try:
        n = float(parts[0])
    except (ValueError, IndexError):
        return 0.0
    unit = (parts[1] if len(parts) > 1 else "MB").upper()
    return round(n / 1024, 1) if unit.startswith("MB") else round(n, 1)


def _speed_mts(s: str) -> int | None:
    for tok in s.split():
        if tok.isdigit():
            return int(tok)
    return None


# --------------------------------------------------------------------------- storage
def disks() -> list[Disk]:
    # JSON output: models like "Samsung SSD 990 EVO 1TB" have spaces, so a positional
    # column split drops them — -J keeps fields intact.
    import json

    rc, out, _ = run(["lsblk", "-dn", "-b", "-J", "-o", "NAME,SIZE,ROTA,MODEL,TRAN,TYPE"], timeout=10)
    if rc != 0 or not out.strip():
        return []
    try:
        devs = json.loads(out).get("blockdevices", [])
    except json.JSONDecodeError:
        return []
    out_disks: list[Disk] = []
    for d in devs:
        name = d.get("name", "")
        if d.get("type") != "disk" or name.startswith(("loop", "zram", "ram")):
            continue
        try:
            size_gb = round(int(d.get("size") or 0) / 1024**3, 1)
        except (ValueError, TypeError):
            size_gb = 0.0
        tran, rota = d.get("tran") or "", str(d.get("rota"))
        if tran == "nvme":
            kind = "NVMe SSD"
        elif rota == "1" or d.get("rota") is True:
            kind = "HDD"
        elif rota == "0" or d.get("rota") is False:
            kind = "SATA SSD" if tran in ("sata", "ata") else "SSD"
        else:
            kind = "disk"
        out_disks.append(Disk(name, size_gb, kind, (d.get("model") or "").strip(), _DISK_CLASS.get(kind)))
    return out_disks


# --------------------------------------------------------------------------- network
def _is_virtual(name: str) -> bool:
    if name == "lo" or name.startswith(("veth", "docker", "br-", "virbr", "vnet", "tap", "tun", "bond")):
        return True
    # No backing device symlink -> virtual/software interface.
    return not (Path("/sys/class/net") / name / "device").exists()


def nics() -> list[NIC]:
    base = Path("/sys/class/net")
    if not base.exists():
        return []
    out: list[NIC] = []
    for nd in sorted(base.iterdir(), key=lambda p: p.name):
        name = nd.name
        if _is_virtual(name):
            continue
        speed = _read_int(nd / "speed")
        state = _read_str(nd / "operstate") or "unknown"
        mac = _read_str(nd / "address")
        out.append(NIC(name, speed if (speed and speed > 0) else None, state, mac))
    return out


def _read_int(p: Path) -> int | None:
    v = _read_str(p)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def _read_str(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


# --------------------------------------------------------------------------- top-level
def host_info(state_dir=None) -> HostInfo:
    return HostInfo(cpu=cpu_info(), mem=mem_info(state_dir), disks=disks(), nics=nics())
