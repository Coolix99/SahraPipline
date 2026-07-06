#!/usr/bin/env python3

import os
import sys
import json
import fnmatch
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import git
from simple_file_checksum import get_checksum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import *
from IO import *


CHECK_META_KEY = "CheckSurface_MetaData"

STATE_TO_META_KEY = {
    "surface": "Surface_MetaData",
    "coord": "Coord_MetaData",
    "thickness": "Thickness_MetaData",
}

LATEST_ORDER = [
    "Thickness_MetaData",
    "Coord_MetaData",
    "Surface_MetaData",
]

SURFACE_FILE_KEYS = [
    "Surface file",
    "Surface file name",
    "surface file",
    "surface file name",
]


def get_git_hash_safe() -> Optional[str]:
    try:
        repo = git.Repo(gitPath, search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return None


def save_metadata_block(folder_path: str, key: str, value: dict) -> None:
    """
    Prefer your pipeline's writeJSON. If it is unavailable or fails,
    fall back to writing MetaData.json directly.
    """
    try:
        writeJSON(folder_path, key, value)
        return
    except Exception as e:
        logging.warning(f"writeJSON failed for {folder_path}; using fallback. Error: {e}")

    metadata_path = os.path.join(folder_path, "MetaData.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[key] = value

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def data_name_from_flatfin_folder(flatfin_folder: str) -> str:
    if flatfin_folder.endswith("_FlatFin"):
        return flatfin_folder[: -len("_FlatFin")]
    return flatfin_folder


def load_name_list(names: Optional[List[str]], names_file: Optional[str]) -> List[str]:
    out = []

    if names:
        out.extend(names)

    if names_file:
        with open(names_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(line)

    return out


def name_matches(folder_name: str, data_name: str, patterns: List[str]) -> bool:
    """
    Accept exact names or shell-style wildcards.
    Examples:
        090626_reg_claudin-gfpxsox10-rfp_pecfin5
        090626_reg_claudin-gfpxsox10-rfp_pecfin5_FlatFin
        *pecfin5*
    """
    if not patterns:
        return True

    for pattern in patterns:
        if pattern == folder_name or pattern == data_name:
            return True
        if fnmatch.fnmatch(folder_name, pattern) or fnmatch.fnmatch(data_name, pattern):
            return True

    return False


def get_surface_file_name(state_metadata: dict) -> Optional[str]:
    for key in SURFACE_FILE_KEYS:
        if key in state_metadata:
            return state_metadata[key]
    return None


def resolve_state(
    metadata: dict,
    requested_state: str,
) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """
    Returns:
        state_meta_key, state_metadata, surface_file_name
    """
    if requested_state == "latest":
        candidate_keys = LATEST_ORDER
    else:
        candidate_keys = [STATE_TO_META_KEY[requested_state]]

    for meta_key in candidate_keys:
        if meta_key not in metadata:
            continue

        state_metadata = metadata[meta_key]
        surface_file_name = get_surface_file_name(state_metadata)

        if surface_file_name is None:
            logging.warning(f"{meta_key} exists but has no surface file key.")
            continue

        return meta_key, state_metadata, surface_file_name

    return None, None, None


def get_surface_checksum_safe(surface_file_path: str) -> Optional[str]:
    try:
        return get_checksum(surface_file_path, algorithm="SHA1")
    except Exception as e:
        logging.warning(f"Could not checksum {surface_file_path}: {e}")
        return None


def already_shown(
    metadata: dict,
    state_meta_key: str,
    surface_file_name: str,
    surface_checksum: Optional[str],
) -> bool:
    check_metadata = metadata.get(CHECK_META_KEY, {})
    shown = check_metadata.get("shown", {})

    record = shown.get(state_meta_key)
    if not record:
        return False

    if record.get("surface file") != surface_file_name:
        return False

    previous_checksum = record.get("surface checksum")

    # If both checksums exist, only skip when they match.
    if surface_checksum is not None and previous_checksum is not None:
        return previous_checksum == surface_checksum

    # Backward-compatible fallback.
    return True


def mark_shown(
    folder_path: str,
    metadata: dict,
    state_meta_key: str,
    surface_file_name: str,
    surface_checksum: Optional[str],
    viewer: str,
) -> None:
    check_metadata = metadata.get(CHECK_META_KEY, {})

    check_metadata["git hash"] = get_git_hash_safe()
    check_metadata["git repo"] = "Sahrapipline"
    check_metadata["last checked"] = datetime.now().isoformat(timespec="seconds")

    shown = check_metadata.get("shown", {})
    shown[state_meta_key] = {
        "shown": True,
        "viewer": viewer,
        "checked at": datetime.now().isoformat(timespec="seconds"),
        "surface file": surface_file_name,
        "surface checksum": surface_checksum,
    }

    check_metadata["shown"] = shown

    save_metadata_block(folder_path, CHECK_META_KEY, check_metadata)


def find_volume_file(volume_root: str, data_name: str) -> Optional[str]:
    """
    Finmask volume is expected at:
        finmasks_path / data_name / data_name.tif

    Falls back to the first tif/tiff inside that folder.
    """
    volume_folder = os.path.join(volume_root, data_name)

    exact_candidates = [
        os.path.join(volume_folder, data_name + ".tif"),
        os.path.join(volume_folder, data_name + ".tiff"),
    ]

    for path in exact_candidates:
        if os.path.exists(path):
            return path

    if not os.path.isdir(volume_folder):
        return None

    tif_files = [
        f for f in os.listdir(volume_folder)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    if not tif_files:
        return None

    tif_files = sorted(tif_files)
    return os.path.join(volume_folder, tif_files[0])


def choose_scalar(mesh: pv.PolyData, preferred_scalar: Optional[str]) -> Optional[str]:
    if preferred_scalar:
        if preferred_scalar in mesh.point_data:
            return preferred_scalar

        logging.warning(
            f"Requested scalar '{preferred_scalar}' not found. "
            f"Available point data: {list(mesh.point_data.keys())}"
        )

    for candidate in [
        "thickness",
        "coord_1",
        "coord_2",
        "mean_curvature",
        "gauss_curvature",
    ]:
        if candidate in mesh.point_data:
            return candidate

    return None


def delete_paths(paths: List[str]) -> None:
    for path in paths:
        if not path:
            continue

        if not os.path.exists(path):
            logging.info(f"Path already gone: {path}")
            continue

        logging.warning(f"Removing: {path}")
        shutil.rmtree(path)


def plot_with_pyvista(
    mesh: pv.PolyData,
    title: str,
    scalar: Optional[str],
    opacity: float,
    show_edges: bool,
) -> str:
    """
    Returns one of:
        "keep"
        "delete"
        "remove"
    """
    action = {"value": "keep"}

    plotter = pv.Plotter()
    plotter.add_text(
        f"{title}\n"
        "q / close: keep + mark shown\n"
        "d: delete FlatFin folder\n"
        "r: remove FlatFin folder + finmask folder",
        position="upper_left",
        font_size=10,
    )

    if scalar is not None:
        plotter.add_mesh(
            mesh,
            scalars=scalar,
            show_edges=show_edges,
            opacity=opacity,
            ambient=0.6,
        )
    else:
        plotter.add_mesh(
            mesh,
            show_edges=show_edges,
            opacity=opacity,
            ambient=0.6,
        )

    def delete_callback():
        action["value"] = "delete"
        plotter.close()

    def remove_callback():
        action["value"] = "remove"
        plotter.close()

    def keep_callback():
        action["value"] = "keep"
        plotter.close()

    plotter.add_key_event("d", delete_callback)
    plotter.add_key_event("r", remove_callback)
    plotter.add_key_event("q", keep_callback)

    plotter.show()

    return action["value"]


def get_napari_surface_data(
    mesh: pv.PolyData,
    scales_zyx: Optional[List[float]],
    scalar: Optional[str],
):
    """
    Napari needs vertices in image coordinates, usually ZYX.

    Preferred:
        mesh.point_data["Coord px"]

    Fallback:
        mesh.points / scales_zyx
    """
    tri_mesh = mesh.triangulate()

    if "Coord px" in tri_mesh.point_data:
        vertices = np.asarray(tri_mesh.point_data["Coord px"], dtype=float)
    elif scales_zyx is not None:
        scales = np.asarray(scales_zyx, dtype=float)
        vertices = np.asarray(tri_mesh.points, dtype=float) / scales
    else:
        vertices = np.asarray(tri_mesh.points, dtype=float)

    faces = tri_mesh.faces.reshape(-1, 4)[:, 1:]

    if scalar is not None and scalar in tri_mesh.point_data:
        values = np.asarray(tri_mesh.point_data[scalar], dtype=float)
        values = np.nan_to_num(values, nan=0.0)
    else:
        values = np.zeros(vertices.shape[0], dtype=float)

    return vertices, faces, values


def plot_with_napari(
    mesh: pv.PolyData,
    volume_path: str,
    title: str,
    scalar: Optional[str],
    scales_zyx: Optional[List[float]],
) -> str:
    """
    Returns one of:
        "keep"
        "delete"
        "remove"
    """
    import napari

    action = {"value": "keep"}

    volume = getImage(volume_path)
    volume_mask = volume > 0

    vertices, faces, values = get_napari_surface_data(
        mesh=mesh,
        scales_zyx=scales_zyx,
        scalar=scalar,
    )

    viewer = napari.Viewer(ndisplay=3, title=title)

    viewer.add_labels(volume_mask.astype(np.uint8), name="finmask volume")
    viewer.add_surface(
        (vertices, faces, values),
        name=f"surface {scalar}" if scalar else "surface",
    )

    @viewer.bind_key("d")
    def delete_callback(viewer):
        action["value"] = "delete"
        viewer.close()

    @viewer.bind_key("r")
    def remove_callback(viewer):
        action["value"] = "remove"
        viewer.close()

    @viewer.bind_key("q")
    def keep_callback(viewer):
        action["value"] = "keep"
        viewer.close()

    print()
    print("Napari keys:")
    print("  q / close : keep + mark shown")
    print("  d         : delete FlatFin folder")
    print("  r         : remove FlatFin folder + finmask folder")
    print()

    napari.run()

    return action["value"]


def get_scales_zyx(metadata: dict) -> Optional[List[float]]:
    for key in [
        "Thickness_MetaData",
        "Coord_MetaData",
        "Surface_MetaData",
        "CenterLine_MetaData",
        "Orient_MetaData",
    ]:
        if key in metadata and "scales ZYX" in metadata[key]:
            return metadata[key]["scales ZYX"]

    return None


def check_one_surface(
    folder_path: str,
    folder_name: str,
    data_name: str,
    requested_state: str,
    viewer: str,
    volume_root: str,
    skip_shown: bool,
    mark_as_shown: bool,
    preferred_scalar: Optional[str],
    opacity: float,
    show_edges: bool,
) -> None:
    metadata = get_JSON(folder_path)

    if not metadata:
        logging.warning(f"Skipping {folder_name}: no metadata found.")
        return

    state_meta_key, state_metadata, surface_file_name = resolve_state(
        metadata,
        requested_state,
    )

    if state_meta_key is None or state_metadata is None or surface_file_name is None:
        logging.warning(
            f"Skipping {folder_name}: could not resolve state '{requested_state}'."
        )
        return

    surface_file_path = os.path.join(folder_path, surface_file_name)

    if not os.path.exists(surface_file_path):
        logging.warning(f"Skipping {folder_name}: missing {surface_file_path}")
        return

    surface_checksum = get_surface_checksum_safe(surface_file_path)

    if skip_shown and already_shown(
        metadata=metadata,
        state_meta_key=state_meta_key,
        surface_file_name=surface_file_name,
        surface_checksum=surface_checksum,
    ):
        logging.info(f"Skipping already shown: {folder_name} [{state_meta_key}]")
        return

    logging.info(f"Loading surface: {surface_file_path}")
    mesh = pv.read(surface_file_path)

    scalar = choose_scalar(mesh, preferred_scalar)

    title = f"{folder_name} | {state_meta_key}"
    if scalar:
        title += f" | scalar: {scalar}"

    volume_path = None
    if viewer == "napari":
        volume_path = find_volume_file(volume_root, data_name)

        if volume_path is None:
            logging.warning(
                f"Skipping {folder_name}: napari mode requested but no finmask volume found."
            )
            return

        logging.info(f"Using volume: {volume_path}")

    if viewer == "pyvista":
        action = plot_with_pyvista(
            mesh=mesh,
            title=title,
            scalar=scalar,
            opacity=opacity,
            show_edges=show_edges,
        )
    elif viewer == "napari":
        scales_zyx = get_scales_zyx(metadata)
        action = plot_with_napari(
            mesh=mesh,
            volume_path=volume_path,
            title=title,
            scalar=scalar,
            scales_zyx=scales_zyx,
        )
    else:
        raise ValueError(f"Unknown viewer: {viewer}")

    finmask_folder = os.path.join(volume_root, data_name)

    if action == "delete":
        delete_paths([folder_path])
        return

    if action == "remove":
        delete_paths([folder_path, finmask_folder])
        return

    if mark_as_shown:
        mark_shown(
            folder_path=folder_path,
            metadata=metadata,
            state_meta_key=state_meta_key,
            surface_file_name=surface_file_name,
            surface_checksum=surface_checksum,
            viewer=viewer,
        )


def check_surfaces(
    surface_dir: str,
    volume_root: str,
    requested_state: str,
    viewer: str,
    names: List[str],
    skip_shown: bool,
    mark_as_shown: bool,
    preferred_scalar: Optional[str],
    opacity: float,
    show_edges: bool,
) -> None:
    surface_folder_list = [
        item for item in os.listdir(surface_dir)
        if os.path.isdir(os.path.join(surface_dir, item))
    ]

    surface_folder_list = sorted(surface_folder_list)

    logging.info(f"Found {len(surface_folder_list)} surface folders.")

    selected = []

    for folder_name in surface_folder_list:
        data_name = data_name_from_flatfin_folder(folder_name)

        if not name_matches(folder_name, data_name, names):
            continue

        selected.append((folder_name, data_name))

    logging.info(f"Selected {len(selected)} folders for checking.")

    for folder_name, data_name in selected:
        folder_path = os.path.join(surface_dir, folder_name)

        logging.info("=" * 80)
        logging.info(f"Checking {folder_name}")

        try:
            check_one_surface(
                folder_path=folder_path,
                folder_name=folder_name,
                data_name=data_name,
                requested_state=requested_state,
                viewer=viewer,
                volume_root=volume_root,
                skip_shown=skip_shown,
                mark_as_shown=mark_as_shown,
                preferred_scalar=preferred_scalar,
                opacity=opacity,
                show_edges=show_edges,
            )
        except Exception as e:
            logging.exception(f"Failed while checking {folder_name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively check FlatFin surfaces."
    )

    parser.add_argument(
        "--surface-dir",
        default=FlatFin_path,
        help="Folder containing FlatFin surface folders. Default: FlatFin_path from config.",
    )

    parser.add_argument(
        "--volume-root",
        default=finmasks_path,
        help="Folder containing finmask volumes. Default: finmasks_path from config.",
    )

    parser.add_argument(
        "--state",
        choices=["latest", "surface", "coord", "thickness"],
        default="latest",
        help=(
            "Which metadata state to show. "
            "'latest' tries thickness, then coord, then surface."
        ),
    )

    parser.add_argument(
        "--viewer",
        choices=["pyvista", "napari"],
        default="pyvista",
        help="pyvista shows only the surface. napari shows surface + finmask volume.",
    )

    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "Optional list of surface/data names or wildcard patterns to show. "
            "Examples: sample1 sample2_FlatFin '*pecfin5*'"
        ),
    )

    parser.add_argument(
        "--names-file",
        default=None,
        help="Optional text file with one surface/data name or wildcard pattern per line.",
    )

    parser.add_argument(
        "--skip-shown",
        dest="skip_shown",
        action="store_true",
        default=True,
        help="Skip surfaces already marked as shown. Default.",
    )

    parser.add_argument(
        "--show-all",
        dest="skip_shown",
        action="store_false",
        help="Do not skip already shown surfaces.",
    )

    parser.add_argument(
        "--no-mark-shown",
        dest="mark_as_shown",
        action="store_false",
        default=True,
        help="Do not write CheckSurface_MetaData after viewing.",
    )

    parser.add_argument(
        "--scalar",
        default=None,
        help=(
            "Point-data scalar to color by. "
            "Default chooses thickness, coord_1, coord_2, mean_curvature, or gauss_curvature if available."
        ),
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=1.0,
        help="Surface opacity for pyvista.",
    )

    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Show mesh edges in pyvista.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    names = load_name_list(args.names, args.names_file)

    check_surfaces(
        surface_dir=args.surface_dir,
        volume_root=args.volume_root,
        requested_state=args.state,
        viewer=args.viewer,
        names=names,
        skip_shown=args.skip_shown,
        mark_as_shown=args.mark_as_shown,
        preferred_scalar=args.scalar,
        opacity=args.opacity,
        show_edges=args.show_edges,
    )