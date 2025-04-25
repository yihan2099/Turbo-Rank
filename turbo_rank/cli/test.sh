#!/usr/bin/env bash
# Save this as scripts/check_cgroup_memory.sh and run `bash scripts/check_cgroup_memory.sh`

pid=$$   # current shell; replace with your torchrun PID if you’re checking a running job
echo "PID: $pid"

# Find the cgroup “sub-path” for the memory controller
cgpath_v2=$(awk -F: '$2=="memory" {print $3}' /proc/$pid/cgroup)
cgpath_v1=$cgpath_v2  # often identical, but v1 is under a different mount

# 1) Check for cgroup v2 (unified)
if [ -f "/sys/fs/cgroup${cgpath_v2}/memory.max" ]; then
  echo "cgroup v2 memory.max:"
  cat "/sys/fs/cgroup${cgpath_v2}/memory.max"
else
  echo "cgroup v2 memory.max not found."
fi

# 2) Check for cgroup v1 (memory subsystem)
if [ -f "/sys/fs/cgroup/memory${cgpath_v1}/memory.limit_in_bytes" ]; then
  echo "cgroup v1 memory.limit_in_bytes:"
  cat "/sys/fs/cgroup/memory${cgpath_v1}/memory.limit_in_bytes"
else
  echo "cgroup v1 memory.limit_in_bytes not found."
fi

# 3) Show all mounts that contain “memory”
echo
echo "Mounts with “memory” in them:"
grep memory /proc/self/mountinfo | awk '{print $5}'