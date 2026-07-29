# controllers

## Identity

Minimal workspace scan — the root directory is `/home/ayu/quadrift/controllers`, suggesting this is the **controllers** sub-package or sub-workspace of a larger project named **quadrift** (likely a quadrotor or quadruped drift/agile-flight platform). No package manifests, source files, or READMEs were returned in the scan, so the sections below reflect only what can be confirmed from the directory path itself.

## Stack

- Parent project: `quadrift`
- Sub-workspace: `controllers`
- Host machine user: `ayu`
- No ROS distro, simulator, or OS confirmed from scan output — these should be filled in once the workspace is explored further.

## Build & Workflow

No build system files (`CMakeLists.txt`, `package.xml`, `setup.py`, `colcon.meta`) were surfaced in the scan. Before working in this workspace, confirm:

```bash
# Check for ROS packages
find /home/ayu/quadrift/controllers -name "package.xml"

# Check build system
ls /home/ayu/quadrift/controllers
ls /home/ayu/quadrift/
```

If this is a colcon workspace, the standard build invocation would be from the workspace root:

```bash
cd /home/ayu/quadrift
colcon build --symlink-install
source install/setup.bash
```

## Debugging

The scan returned no file contents. If future sessions also return sparse results, check:

```bash
# Confirm directory is populated
find /home/ayu/quadrift/controllers -type f | head -40

# Look for any existing docs
find /home/ayu/quadrift -name "README*" -o -name "AGENTS.md" -o -name "CLAUDE.md" 2>/dev/null
```

This DRIFT.md should be regenerated once the workspace contents are accessible — the current scan did not return enough grounded detail to document conventions, topics, frames, or performance targets.