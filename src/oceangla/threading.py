import os
from pathlib import Path

def get_optimal_workers(sample_path: str | Path | None = None, is_compressed: bool = True) -> int:
    """
    Choose a safe default max_workers for concurrent NIfTI loading based on
    available CPU cores, file compression, and whether the path is on a network drive.
    """
    # 1. Respect Linux cgroup / Slurm / taskset CPU affinity if available, else fall back to os.cpu_count()
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 4

    # 2. Detect if files reside on a network filesystem (NFS, SMB, Lustre, GPFS, etc.)
    is_network_fs = False
    if sample_path is not None and os.path.exists("/proc/mounts"):
        target = os.path.realpath(sample_path)
        best_mount_len = -1
        network_fs_types = {"nfs", "nfs4", "cifs", "smb", "smbfs", "lustre", "gpfs", "fuse.sshfs"}
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mount_point, fs_type = parts[1], parts[2]
                    if target.startswith(mount_point) and len(mount_point) > best_mount_len:
                        best_mount_len = len(mount_point)
                        is_network_fs = fs_type in network_fs_types

    # 3. Apply workload-aware caps
    if is_network_fs:
        # Network drives choke on high concurrency due to metadata RPC contention
        return min(cpus, 8)
    elif is_compressed:
        # .nii.gz decompression is mostly CPU-bound (zlib); match physical/available cores
        return min(cpus, 16)
    else:
        # Uncompressed .nii on local SSD/NVMe is pure I/O; slight oversubscription helps
        return min(cpus * 2, 32)