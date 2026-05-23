import json
import math
import os
import random
from dataclasses import dataclass

import carb
import omni.usd
import omni.anim.navigation.core as nav
from pxr import UsdGeom, Sdf, Gf


@dataclass
class NavInitCfg:
    enabled: bool = True
    output_json: str = "generated_paths/nav_path.json"
    robot_candidates: tuple[str, ...] = (
        "/World/map/robot_model",
        "/World/map/robot_model/uwc",
        "/World/map/robot_model/UWC",
        "/World/robot_model/uwc",
        "/World/robot_model/UWC",
        "/World/uwc",
        "/World/UWC",
        "/World/robot_model",
    )
    min_start_goal_dist: float = 40.0
    max_pair_try: int = 700
    max_sample_try: int = 3000

    z_offset: float = 0.15
    path_width: float = 0.12
    floor_z_min: float = -0.10
    floor_z_max: float = 0.30
    spawn_z_offset: float = 0.5

    draw_debug: bool = True
    save_json: bool = True


def _dist2d(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _get_stage_and_navmesh():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("[NAV_INIT] No opened stage.")

    inav = nav.acquire_interface()
    navmesh = inav.get_navmesh()
    if navmesh is None:
        raise RuntimeError("[NAV_INIT] NavMesh is None. Enable navigation extension and bake navmesh first.")
    return stage, inav, navmesh


def _collect_floor_triangles(inav, navmesh, cfg: NavInitCfg):
    walk_idx = inav.find_area("Walkable")
    tris = navmesh.get_draw_triangles(walk_idx)
    if len(tris) == 0:
        raise RuntimeError("[NAV_INIT] No Walkable triangles in NavMesh.")

    floor_tri_bases = []
    for i in range(0, len(tris), 3):
        a, b, c = tris[i], tris[i + 1], tris[i + 2]
        zavg = (float(a[2]) + float(b[2]) + float(c[2])) / 3.0
        if cfg.floor_z_min <= zavg <= cfg.floor_z_max:
            floor_tri_bases.append(i)

    if len(floor_tri_bases) == 0:
        raise RuntimeError("[NAV_INIT] No floor-like nav triangles. Tune floor_z_min/max.")
    return walk_idx, tris, floor_tri_bases


def _make_area_costs(inav, walk_idx):
    try:
        area_count = inav.get_area_count()
        area_costs = [-1.0] * area_count
        area_costs[walk_idx] = 1.0
        return area_costs
    except Exception:
        return None


def _sample_random_navmesh_point(navmesh, walk_idx, tris, floor_tri_bases, cfg: NavInitCfg, required_island=None):
    for _ in range(cfg.max_sample_try):
        tri_base = random.choice(floor_tri_bases)
        a, b, c = tris[tri_base], tris[tri_base + 1], tris[tri_base + 2]

        r1 = math.sqrt(random.random())
        r2 = random.random()

        x = (1.0 - r1) * float(a[0]) + r1 * (1.0 - r2) * float(b[0]) + r1 * r2 * float(c[0])
        y = (1.0 - r1) * float(a[1]) + r1 * (1.0 - r2) * float(b[1]) + r1 * r2 * float(c[1])
        z = (1.0 - r1) * float(a[2]) + r1 * (1.0 - r2) * float(b[2]) + r1 * r2 * float(c[2])

        raw = carb.Float3(x, y, z)

        if required_island is None:
            point, island = navmesh.query_closest_point(
                target=raw, area_indices=[walk_idx], search_island_id=-1
            )
        else:
            point, island = navmesh.query_closest_point(
                target=raw, area_indices=[walk_idx], search_island_id=required_island
            )

        if point is not None:
            return point, island

    return None, -1


def _compute_path(navmesh, area_costs, start_point, goal_point):
    try:
        if area_costs is not None:
            return navmesh.query_shortest_path(
                start_pos=carb.Float3(float(start_point[0]), float(start_point[1]), float(start_point[2])),
                end_pos=carb.Float3(float(goal_point[0]), float(goal_point[1]), float(goal_point[2])),
                area_costs=area_costs,
                straighten=False,
            )
        return navmesh.query_shortest_path(start_point, goal_point)
    except Exception:
        return None


def _get_path_points(path):
    try:
        return list(path.get_points())
    except Exception:
        try:
            return list(path)
        except Exception:
            return []


def _generate_random_path(navmesh, walk_idx, tris, floor_tri_bases, area_costs, cfg: NavInitCfg):
    for _ in range(cfg.max_pair_try):
        sp, si = _sample_random_navmesh_point(navmesh, walk_idx, tris, floor_tri_bases, cfg, required_island=None)
        if sp is None:
            continue

        gp, _ = _sample_random_navmesh_point(navmesh, walk_idx, tris, floor_tri_bases, cfg, required_island=si)
        if gp is None:
            continue

        if _dist2d(sp, gp) < cfg.min_start_goal_dist:
            continue

        path = _compute_path(navmesh, area_costs, sp, gp)
        if path is None:
            continue

        points = _get_path_points(path)
        if len(points) >= 2:
            return sp, gp, points

    raise RuntimeError("[NAV_INIT] Random path generation failed. Lower min_start_goal_dist.")


def _spawn_robot_at_path_start(stage, start_point, points, cfg: NavInitCfg):
    robot_prim = None
    robot_path = None
    for path in cfg.robot_candidates:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            robot_prim = prim
            robot_path = path
            break

    if robot_prim is None:
        raise RuntimeError(f"[NAV_INIT] Robot root prim not found. candidates={cfg.robot_candidates}")

    if len(points) >= 2:
        p0, p1 = points[0], points[1]
        yaw_rad = math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0]))
    else:
        yaw_rad = 0.0
    yaw_deg = math.degrees(yaw_rad)

    desired_world_pos = Gf.Vec3d(float(start_point[0]), float(start_point[1]), float(start_point[2]) + cfg.spawn_z_offset)

    parent_prim = robot_prim.GetParent()
    cache = UsdGeom.XformCache()
    parent_world_tf = cache.GetLocalToWorldTransform(parent_prim)
    desired_local_pos = parent_world_tf.GetInverse().Transform(desired_world_pos)

    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(desired_local_pos)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))

    print(f"[NAV_INIT] Robot spawned at {robot_path}, yaw={yaw_deg:.2f}")


def _draw_path_debug(stage, start_point, goal_point, points, cfg: NavInitCfg):
    for p in ("/World/NavDebug_PathLine", "/World/NavDebug_StartMarker", "/World/NavDebug_GoalMarker"):
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            stage.RemovePrim(p)

    curve = UsdGeom.BasisCurves.Define(stage, Sdf.Path("/World/NavDebug_PathLine"))
    curve_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]) + cfg.z_offset) for p in points]
    curve.CreatePointsAttr(curve_points)
    curve.CreateCurveVertexCountsAttr([len(curve_points)])
    curve.CreateWidthsAttr([cfg.path_width])
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 0.0)])

    def _marker(path_str, pos, radius, color):
        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path_str))
        sphere.CreateRadiusAttr(radius)
        sphere.CreateDisplayColorAttr([color])
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]) + cfg.z_offset))

    _marker("/World/NavDebug_StartMarker", start_point, 1.0, Gf.Vec3f(0.0, 1.0, 0.0))
    _marker("/World/NavDebug_GoalMarker", goal_point, 1.0, Gf.Vec3f(1.0, 0.0, 0.0))

    print("[NAV_INIT] Debug path drawn.")


def _save_path_json(output_path, start_point, goal_point, points):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "frame_id": "odom",
        "start": {"x": float(start_point[0]), "y": float(start_point[1]), "z": float(start_point[2])},
        "goal": {"x": float(goal_point[0]), "y": float(goal_point[1]), "z": float(goal_point[2])},
        "points": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in points],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[NAV_INIT] Saved path json: {output_path}")


def run_nav_init(cfg: NavInitCfg | None = None):
    cfg = cfg or NavInitCfg()
    if not cfg.enabled:
        print("[NAV_INIT] disabled.")
        return None

    stage, inav, navmesh = _get_stage_and_navmesh()
    walk_idx, tris, floor_tri_bases = _collect_floor_triangles(inav, navmesh, cfg)
    area_costs = _make_area_costs(inav, walk_idx)
    start_point, goal_point, points = _generate_random_path(navmesh, walk_idx, tris, floor_tri_bases, area_costs, cfg)
    _spawn_robot_at_path_start(stage, start_point, points, cfg)

    if cfg.draw_debug:
        _draw_path_debug(stage, start_point, goal_point, points, cfg)
    if cfg.save_json:
        _save_path_json(cfg.output_json, start_point, goal_point, points)

    print("[NAV_INIT] done.")
    return start_point, goal_point, points