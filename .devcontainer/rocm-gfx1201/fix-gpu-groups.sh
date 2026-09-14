#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# fix-gpu-groups.sh
#
# Detects the ACTUAL host GID that owns /dev/kfd and /dev/dri/renderD* inside
# this container (these are bind-mounted device nodes; their owning GID is
# whatever udev assigned on the HOST, which varies per machine and can't be
# baked into the image at build time), creates/reuses a matching group, and
# adds the target user to it.
#
# Must be run as root. Safe to run multiple times (idempotent).
#
# Called by devcontainer.json's postStartCommand on every container start.
# VS Code's Dev Containers CLI overrides the image's ENTRYPOINT/CMD with its
# own keep-alive wrapper on every container it manages
# ------------------------------------------------------------------------------

TARGET_USER="${AETHER_CONTAINER_USER:-ubuntu}"

if [ "$(id -u)" -ne 0 ]; then
    echo "[fix-gpu-groups] Must be run as root (got uid=$(id -u)). Aborting." >&2
    exit 1
fi

ensure_group_for_device() {
    local device_path="$1"
    local fallback_name="$2"

    if [ ! -e "${device_path}" ]; then
        echo "[fix-gpu-groups] Warning: ${device_path} not found (GPU device not passed through?) - skipping" >&2
        return
    fi

    local device_gid
    device_gid="$(stat -c '%g' "${device_path}")"

    # Reuse a group already owning this GID; only create one if none exists.
    local group_name
    group_name="$(getent group "${device_gid}" | cut -d: -f1 || true)"

    if [ -z "${group_name}" ]; then
        group_name="${fallback_name}"
        # fallback_name may already exist under a different GID (e.g. leftover
        # from the image build stage) - avoid a groupadd collision if so.
        if getent group "${group_name}" > /dev/null 2>&1; then
            group_name="${fallback_name}_${device_gid}"
        fi
        groupadd -g "${device_gid}" "${group_name}"
        echo "[fix-gpu-groups] Created group '${group_name}' (gid=${device_gid}) for ${device_path}"
    else
        echo "[fix-gpu-groups] Reusing existing group '${group_name}' (gid=${device_gid}) for ${device_path}"
    fi

    # Only usermod if not already a member - keeps repeated runs cheap/quiet.
    if ! id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx "${group_name}"; then
        usermod -aG "${group_name}" "${TARGET_USER}"
    fi
}

ensure_group_for_device "/dev/kfd" "kfd"

# /dev/dri can expose multiple render nodes (renderD128, renderD129, ...);
# resolve whatever is actually present rather than assuming renderD128.
if [ -d /dev/dri ]; then
    shopt -s nullglob
    render_nodes=(/dev/dri/renderD*)
    shopt -u nullglob
    if [ "${#render_nodes[@]}" -eq 0 ]; then
        echo "[fix-gpu-groups] Warning: no /dev/dri/renderD* nodes found - skipping" >&2
    fi
    for node in "${render_nodes[@]}"; do
        ensure_group_for_device "${node}" "render"
    done
else
    echo "[fix-gpu-groups] Warning: /dev/dri not found - skipping render node group setup" >&2
fi

# Hard gate: if the GPU *was* passed through, the user must have ended up in the
# group owning /dev/kfd. Build-time group setup no longer papers over a miss here.
if [ -e /dev/kfd ]; then
    kfd_gid="$(stat -c '%g' /dev/kfd)"
    if ! id -G "${TARGET_USER}" | tr ' ' '\n' | grep -qx "${kfd_gid}"; then
        echo "[fix-gpu-groups] FATAL: ${TARGET_USER} is not in the group owning /dev/kfd (gid=${kfd_gid})." >&2
        echo "[fix-gpu-groups] Groups: $(id -nG "${TARGET_USER}")" >&2
        exit 1
    fi
fi

echo "[fix-gpu-groups] Final group membership for ${TARGET_USER}: $(id -nG "${TARGET_USER}")"
