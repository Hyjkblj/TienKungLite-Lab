import os
import shutil
import struct
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "mjcf"
OUTPUT_PATH = OUTPUT_DIR / "real_lite.xml"
GENERATED_MESH_DIR = OUTPUT_DIR / "_mesh_cache"
ASSET_ROOT_ENV_VAR = "TIENKUNG_LITE_ASSET_ROOT"
ASSET_ROOT_DIRNAME = "x_humanoid_0430_newfeet_newbody_publish"
DEFAULT_REAL_LITE_ASSET_ROOT = ROOT.parent / "lite_urdf_publish" / ASSET_ROOT_DIRNAME
FALLBACK_REAL_LITE_ASSET_ROOT = ROOT.parent / ASSET_ROOT_DIRNAME


def _missing_required_entries(asset_root: Path) -> list[str]:
    missing_entries = []
    for required_dir in ("meshes", "urdf"):
        if not (asset_root / required_dir).is_dir():
            missing_entries.append(required_dir)
    return missing_entries


def _default_asset_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for candidate in (DEFAULT_REAL_LITE_ASSET_ROOT, FALLBACK_REAL_LITE_ASSET_ROOT):
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in candidates:
            candidates.append(resolved_candidate)
    return tuple(candidates)


AUTO_DISCOVERED_ASSET_ROOTS = _default_asset_roots()


def resolve_real_lite_asset_root() -> Path:
    configured_path = os.getenv(ASSET_ROOT_ENV_VAR)
    if configured_path:
        asset_root = Path(configured_path).expanduser().resolve()
        if not asset_root.exists():
            raise FileNotFoundError(
                f"Real Lite assets not found at: {asset_root}\n"
                f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
            )

        missing_entries = _missing_required_entries(asset_root)
        if missing_entries:
            raise FileNotFoundError(
                f"Real Lite asset root is missing required directories {missing_entries!r}: {asset_root}\n"
                f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
            )
        return asset_root

    searched_roots = []
    for asset_root in AUTO_DISCOVERED_ASSET_ROOTS:
        searched_roots.append(asset_root)
        if asset_root.exists() and not _missing_required_entries(asset_root):
            return asset_root

    searched_root_lines = "\n".join(f"  - {candidate}" for candidate in searched_roots)
    raise FileNotFoundError(
        "Real Lite assets were not found in any default location.\n"
        f"Searched:\n{searched_root_lines}\n"
        f"Set {ASSET_ROOT_ENV_VAR} to the external asset directory that contains 'meshes' and 'urdf'."
    )


ASSET_ROOT = resolve_real_lite_asset_root()
URDF_PATH = ASSET_ROOT / "urdf" / "humanoid_publish.urdf"
MESH_DIR = ASSET_ROOT / "meshes"


def _meshdir_for_xml() -> str:
    relative_mesh_dir = os.path.relpath(GENERATED_MESH_DIR, OUTPUT_DIR)
    return Path(relative_mesh_dir).as_posix().rstrip("/") + "/"


def _is_valid_binary_stl_bytes(stl_bytes: bytes) -> bool:
    if len(stl_bytes) < 84:
        return False

    triangle_data_length = len(stl_bytes) - 84
    if triangle_data_length <= 0 or triangle_data_length % 50 != 0:
        return False

    expected_triangle_count = triangle_data_length // 50
    declared_triangle_count = struct.unpack("<I", stl_bytes[80:84])[0]
    return declared_triangle_count == expected_triangle_count


def _parse_ascii_stl(stl_bytes: bytes, mesh_path: Path) -> list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]]:
    triangles: list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]] = []
    current_normal: tuple[float, float, float] | None = None
    current_vertices: list[tuple[float, float, float]] = []

    decoded_text = stl_bytes.decode("utf-8", errors="ignore")
    for raw_line in decoded_text.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        tokens = line.split()
        if not tokens:
            continue

        keyword = tokens[0].lower()
        if keyword == "facet" and len(tokens) >= 5 and tokens[1].lower() == "normal":
            current_normal = (float(tokens[2]), float(tokens[3]), float(tokens[4]))
            current_vertices = []
        elif keyword == "vertex" and len(tokens) >= 4:
            current_vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        elif keyword == "endfacet":
            if current_normal is None or len(current_vertices) != 3:
                raise ValueError(f"Malformed ASCII STL facet in {mesh_path}")
            triangles.append((current_normal, current_vertices.copy()))
            current_normal = None
            current_vertices = []

    if not triangles:
        raise ValueError(f"No triangles found in ASCII STL file: {mesh_path}")
    return triangles


def _write_binary_stl(
    output_path: Path,
    triangles: list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]],
) -> None:
    header = b"TienKungLiteLab generated binary STL".ljust(80, b"\0")
    with output_path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(triangles)))
        for normal, vertices in triangles:
            packed = list(normal)
            for vertex in vertices:
                packed.extend(vertex)
            handle.write(struct.pack("<12f", *packed))
            handle.write(struct.pack("<H", 0))


def _prepare_mujoco_mesh_cache() -> None:
    GENERATED_MESH_DIR.mkdir(parents=True, exist_ok=True)
    for mesh_path in MESH_DIR.iterdir():
        if not mesh_path.is_file():
            continue
        output_path = GENERATED_MESH_DIR / mesh_path.name
        if mesh_path.suffix.lower() != ".stl":
            shutil.copy2(mesh_path, output_path)
            continue

        stl_bytes = mesh_path.read_bytes()
        if _is_valid_binary_stl_bytes(stl_bytes):
            shutil.copy2(mesh_path, output_path)
            continue

        triangles = _parse_ascii_stl(stl_bytes, mesh_path)
        _write_binary_stl(output_path, triangles)

# MuJoCo fullinertia order is:
#   Ixx Iyy Izz Ixy Ixz Iyz
CRITICAL_FULLINERTIA_LINKS = (
    "pelvis",
    "hip_pitch_l_link",
    "knee_pitch_l_link",
    "hip_pitch_r_link",
    "knee_pitch_r_link",
    "waist_link",
)
CRITICAL_MASS_LINKS = ("waist_link",)
INERTIA_TOLERANCE = 1e-9
MASS_TOLERANCE = 1e-6

# WARNING: This MJCF is hand-converted from the external URDF asset directory.
# If the URDF changes (joint ranges, meshes, inertials, collision geometry),
# this MJCF MUST be updated manually to match. sim2sim results are only
# valid when MJCF and URDF agree on joint limits and dynamics.

MJCF_TEXT = f"""<mujoco model="real_lite">
  <option gravity="0 0 -9.81" solver="PGS"/>
  <option integrator="implicitfast"/>
  <size njmax="500" nconmax="100"/>
  <compiler angle="radian" meshdir="{_meshdir_for_xml()}" eulerseq="zyx"/>

  <default>
    <joint type="hinge" limited="true"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    <mesh name="pelvis" file="pelvis.STL"/>
    <mesh name="waist_link" file="waist_link.STL"/>
    <mesh name="hip_roll_l_link" file="hip_roll_l_link.STL"/>
    <mesh name="hip_yaw_l_link" file="hip_yaw_l_link.STL"/>
    <mesh name="hip_pitch_l_link" file="hip_pitch_l_link.STL"/>
    <mesh name="knee_pitch_l_link" file="knee_pitch_l_link.STL"/>
    <mesh name="ankle_pitch_l_link" file="ankle_pitch_l_link.STL"/>
    <mesh name="ankle_roll_l_link" file="ankle_roll_l_link.STL"/>
    <mesh name="hip_roll_r_link" file="hip_roll_r_link.STL"/>
    <mesh name="hip_yaw_r_link" file="hip_yaw_r_link.STL"/>
    <mesh name="hip_pitch_r_link" file="hip_pitch_r_link.STL"/>
    <mesh name="knee_pitch_r_link" file="knee_pitch_r_link.STL"/>
    <mesh name="ankle_pitch_r_link" file="ankle_pitch_r_link.STL"/>
    <mesh name="ankle_roll_r_link" file="ankle_roll_r_link.STL"/>
    <mesh name="shoulder_pitch_l_link" file="shoulder_pitch_l_link.STL"/>
    <mesh name="shoulder_roll_l_link" file="shoulder_roll_l_link.STL"/>
    <mesh name="shoulder_yaw_l_link" file="shoulder_yaw_l_link.STL"/>
    <mesh name="elbow_l_link" file="elbow_l_link.STL"/>
    <mesh name="shoulder_pitch_r_link" file="shoulder_pitch_r_link.STL"/>
    <mesh name="shoulder_roll_r_link" file="shoulder_roll_r_link.STL"/>
    <mesh name="shoulder_yaw_r_link" file="shoulder_yaw_r_link.STL"/>
    <mesh name="elbow_r_link" file="elbow_r_link.STL"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
    <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" pos="0 0 0" size="100 100 1" type="plane" material="matplane" margin="0.001" contype="1" conaffinity="15" friction="1 0.005 0.0001"/>

    <body name="pelvis" pos="0 0 1.0">
      <freejoint name="root"/>
      <inertial pos="-0.047395 0.0 -0.046657" mass="5.587228" fullinertia="0.080339 0.029082 0.091570 -0.0000080 0.006666 -0.000004"/>
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="pelvis"/>
      <geom size="0.085 0.055" pos="-0.02 0 -0.11" quat="0.707388 0.706825 0 0" type="cylinder" rgba="0.75 0.75 0.75 1"/>
      <site name="imu" size="0.01" pos="0 0 0"/>

      <body name="hip_roll_l_link" pos="0 0.13 -0.079">
        <inertial pos="-0.00048 0.00336 -0.00372" mass="1.12809" diaginertia="0.00058 0.00093 0.00108"/>
        <joint name="hip_roll_l_joint" pos="0 0 0" axis="1 0 0" range="-0.97 0.97" damping="10"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="hip_roll_l_link"/>
        <body name="hip_yaw_l_link" pos="0 0 -0.116">
          <inertial pos="0.00104 0.00351 0.00968" mass="1.91605" diaginertia="0.00237 0.00271 0.00152"/>
          <joint name="hip_yaw_l_joint" pos="0 0 0" axis="0 0 1" range="-1.0472 1.0472" damping="5"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="hip_yaw_l_link"/>
          <body name="hip_pitch_l_link">
            <inertial pos="0.00823 -0.00853 -0.13551" mass="3.37311" fullinertia="0.02261 0.02274 0.00637 0.00036 -0.00020 -0.00132"/>
            <joint name="hip_pitch_l_joint" pos="0 0 0" axis="0 1 0" range="-1.57 0.5236" damping="10"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="hip_pitch_l_link"/>
            <body name="knee_pitch_l_link" pos="0 0 -0.3">
              <inertial pos="0.00228 0.00293 -0.12066" mass="2.28832" fullinertia="0.02033 0.02013 0.00090 0.00004 0.00049 0.00089"/>
              <joint name="knee_pitch_l_joint" pos="0 0 0" axis="0 1 0" range="0.1745 2.443" damping="10"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="knee_pitch_l_link"/>
              <body name="ankle_pitch_l_link" pos="0 0 -0.3">
                <inertial pos="0.00027 0.0 0.0" mass="0.15163" diaginertia="0.00003 0.00003 0.00006"/>
                <joint name="ankle_pitch_l_joint" pos="0 0 0" axis="0 1 0" range="-1.22 0.5236" damping="2.5"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="ankle_pitch_l_link"/>
                <body name="ankle_roll_l_link">
                  <inertial pos="0.004998 0 -0.026936" mass="0.6583335" diaginertia="0.00051357 0.0021761 0.0023671"/>
                  <joint name="ankle_roll_l_joint" pos="0 0 0" axis="1 0 0" range="-0.4363 0.4363" damping="1.4"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="ankle_roll_l_link"/>
                  <!-- Use a full sole contact patch instead of two narrow toe rails.
                       This better matches the support polygon Isaac sees and reduces
                       pitch/roll instability in MuJoCo stand tests. -->
                  <geom name="sole_left" contype="2" conaffinity="1" size="0.115 0.040 0.015" pos="0.035 0 -0.042" type="box" rgba="0.75 0.75 0.75 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <body name="hip_roll_r_link" pos="0 -0.13 -0.079">
        <inertial pos="-0.00048 -0.00336 -0.00372" mass="1.12809" diaginertia="0.00058 0.00093 0.00108"/>
        <joint name="hip_roll_r_joint" pos="0 0 0" axis="1 0 0" range="-0.97 0.97" damping="10"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="hip_roll_r_link"/>
        <body name="hip_yaw_r_link" pos="0 0 -0.116">
          <inertial pos="0.00104 -0.00351 0.00968" mass="1.91605" diaginertia="0.00237 0.00271 0.00152"/>
          <joint name="hip_yaw_r_joint" pos="0 0 0" axis="0 0 1" range="-1.0472 1.0472" damping="5"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="hip_yaw_r_link"/>
          <body name="hip_pitch_r_link">
            <inertial pos="0.00823 0.00853 -0.13551" mass="3.37311" fullinertia="0.02261 0.02274 0.00637 -0.00036 -0.00020 0.00132"/>
            <joint name="hip_pitch_r_joint" pos="0 0 0" axis="0 1 0" range="-1.57 0.5236" damping="10"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="hip_pitch_r_link"/>
            <body name="knee_pitch_r_link" pos="0 0 -0.3">
              <inertial pos="0.00228 -0.00293 -0.12066" mass="2.28832" fullinertia="0.02033 0.02013 0.00090 -0.00004 0.00049 -0.00089"/>
              <joint name="knee_pitch_r_joint" pos="0 0 0" axis="0 1 0" range="0.1745 2.443" damping="10"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="knee_pitch_r_link"/>
              <body name="ankle_pitch_r_link" pos="0 0 -0.3">
                <inertial pos="0.00027 0.0 0.0" mass="0.15163" diaginertia="0.00003 0.00003 0.00006"/>
                <joint name="ankle_pitch_r_joint" pos="0 0 0" axis="0 1 0" range="-1.22 0.5236" damping="2.5"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="ankle_pitch_r_link"/>
                <body name="ankle_roll_r_link">
                  <inertial pos="0.004998 0 -0.026936" mass="0.6583335" diaginertia="0.00051357 0.0021761 0.0023671"/>
                  <joint name="ankle_roll_r_joint" pos="0 0 0" axis="1 0 0" range="-0.4363 0.4363" damping="1.4"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="ankle_roll_r_link"/>
                  <geom name="sole_right" contype="2" conaffinity="1" size="0.115 0.040 0.015" pos="0.035 0 -0.042" type="box" rgba="0.75 0.75 0.75 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>

      <body name="waist_link" pos="0 0 0.0192">
        <inertial pos="-0.005467 -0.000016 0.302017" mass="13.8" fullinertia="0.527626 0.471407 0.111941 0.000095 -0.001701 -0.000322"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="waist_link"/>

        <body name="shoulder_pitch_l_link" pos="0 0.17581 0.43652" quat="0.991445 0.130526 0 0">
          <inertial pos="0.000988 0.030202 0.000417" mass="0.163757" diaginertia="0.000156 0.000183 0.000224"/>
          <joint name="shoulder_pitch_l_joint" pos="0 0 0" axis="0 1 0" range="-3.14 0.97" damping="3"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="shoulder_pitch_l_link"/>
          <body name="shoulder_roll_l_link" pos="-0.0025 0.062 0">
            <inertial pos="0.004522 -0.000011 -0.042682" mass="0.933844" diaginertia="0.001705 0.00172 0.000234"/>
            <joint name="shoulder_roll_l_joint" pos="0 0 0" axis="1 0 0" range="-0.08 3.49" damping="1.5"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="shoulder_roll_l_link"/>
            <body name="shoulder_yaw_l_link" pos="0 0 -0.107">
              <inertial pos="-0.000044 -0.002833 -0.092643" mass="0.610091" diaginertia="0.000943 0.000918 0.00022"/>
              <joint name="shoulder_yaw_l_joint" pos="0 0 0" axis="0 0 1" range="-2.96 2.96" damping="1"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="shoulder_yaw_l_link"/>
              <body name="elbow_l_link" pos="0 0 -0.11">
                <inertial pos="-0.000267 -0.001013 -0.14593" mass="0.341159" diaginertia="0.005196 0.005151 0.00022"/>
                <joint name="elbow_l_joint" pos="0 0 0" axis="0 1 0" range="-2.1 0.0" damping="1"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.9 0.92 0.93 1" mesh="elbow_l_link"/>
              </body>
            </body>
          </body>
        </body>

        <body name="shoulder_pitch_r_link" pos="0 -0.17577 0.43652" quat="0.991445 -0.130526 0 0">
          <inertial pos="0.000988 -0.030202 0.000417" mass="0.163757" diaginertia="0.000156 0.000183 0.000224"/>
          <joint name="shoulder_pitch_r_joint" pos="0 0 0" axis="0 1 0" range="-3.14 0.97" damping="3"/>
          <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="shoulder_pitch_r_link"/>
          <body name="shoulder_roll_r_link" pos="-0.0025 -0.062 0">
            <inertial pos="0.004522 0.000011 -0.042682" mass="0.933844" diaginertia="0.001705 0.00172 0.000234"/>
            <joint name="shoulder_roll_r_joint" pos="0 0 0" axis="1 0 0" range="-3.49 0.08" damping="1.5"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="shoulder_roll_r_link"/>
            <body name="shoulder_yaw_r_link" pos="0 0 -0.107">
              <inertial pos="-0.000044 0.002833 -0.092643" mass="0.610091" diaginertia="0.000943 0.000918 0.00022"/>
              <joint name="shoulder_yaw_r_joint" pos="0 0 0" axis="0 0 1" range="-2.96 2.96" damping="1"/>
              <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75 0.75 0.75 1" mesh="shoulder_yaw_r_link"/>
              <body name="elbow_r_link" pos="0 0 -0.11">
                <inertial pos="-0.000267 0.001013 -0.14593" mass="0.341159" diaginertia="0.005196 0.005151 0.00022"/>
                <joint name="elbow_r_joint" pos="0 0 0" axis="0 1 0" range="-2.1 0.0" damping="1"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.9 0.92 0.93 1" mesh="elbow_r_link"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="hip_roll_l_joint1" joint="hip_roll_l_joint" kp="700"/>
    <position name="hip_yaw_l_joint1" joint="hip_yaw_l_joint" kp="500"/>
    <position name="hip_pitch_l_joint1" joint="hip_pitch_l_joint" kp="700"/>
    <position name="knee_pitch_l_joint1" joint="knee_pitch_l_joint" kp="700"/>
    <position name="ankle_pitch_l_joint1" joint="ankle_pitch_l_joint" kp="30"/>
    <position name="ankle_roll_l_joint1" joint="ankle_roll_l_joint" kp="16.8"/>
    <position name="hip_roll_r_joint1" joint="hip_roll_r_joint" kp="700"/>
    <position name="hip_yaw_r_joint1" joint="hip_yaw_r_joint" kp="500"/>
    <position name="hip_pitch_r_joint1" joint="hip_pitch_r_joint" kp="700"/>
    <position name="knee_pitch_r_joint1" joint="knee_pitch_r_joint" kp="700"/>
    <position name="ankle_pitch_r_joint1" joint="ankle_pitch_r_joint" kp="30"/>
    <position name="ankle_roll_r_joint1" joint="ankle_roll_r_joint" kp="16.8"/>
    <position name="shoulder_pitch_l_joint1" joint="shoulder_pitch_l_joint" kp="60"/>
    <position name="shoulder_roll_l_joint1" joint="shoulder_roll_l_joint" kp="20"/>
    <position name="shoulder_yaw_l_joint1" joint="shoulder_yaw_l_joint" kp="10"/>
    <position name="elbow_l_joint1" joint="elbow_l_joint" kp="10"/>
    <position name="shoulder_pitch_r_joint1" joint="shoulder_pitch_r_joint" kp="60"/>
    <position name="shoulder_roll_r_joint1" joint="shoulder_roll_r_joint" kp="20"/>
    <position name="shoulder_yaw_r_joint1" joint="shoulder_yaw_r_joint" kp="10"/>
    <position name="elbow_r_joint1" joint="elbow_r_joint" kp="10"/>
  </actuator>

  <sensor>
    <jointpos joint="hip_roll_l_joint" user="6"/>
    <jointpos joint="hip_pitch_l_joint" user="6"/>
    <jointpos joint="hip_yaw_l_joint" user="6"/>
    <jointpos joint="knee_pitch_l_joint" user="6"/>
    <jointpos joint="ankle_pitch_l_joint" user="6"/>
    <jointpos joint="ankle_roll_l_joint" user="6"/>
    <jointpos joint="hip_roll_r_joint" user="6"/>
    <jointpos joint="hip_pitch_r_joint" user="6"/>
    <jointpos joint="hip_yaw_r_joint" user="6"/>
    <jointpos joint="knee_pitch_r_joint" user="6"/>
    <jointpos joint="ankle_pitch_r_joint" user="6"/>
    <jointpos joint="ankle_roll_r_joint" user="6"/>
    <jointpos joint="shoulder_pitch_l_joint" user="6"/>
    <jointpos joint="shoulder_roll_l_joint" user="6"/>
    <jointpos joint="shoulder_yaw_l_joint" user="6"/>
    <jointpos joint="elbow_l_joint" user="6"/>
    <jointpos joint="shoulder_pitch_r_joint" user="6"/>
    <jointpos joint="shoulder_roll_r_joint" user="6"/>
    <jointpos joint="shoulder_yaw_r_joint" user="6"/>
    <jointpos joint="elbow_r_joint" user="6"/>

    <jointvel joint="hip_roll_l_joint" user="6"/>
    <jointvel joint="hip_pitch_l_joint" user="6"/>
    <jointvel joint="hip_yaw_l_joint" user="6"/>
    <jointvel joint="knee_pitch_l_joint" user="6"/>
    <jointvel joint="ankle_pitch_l_joint" user="6"/>
    <jointvel joint="ankle_roll_l_joint" user="6"/>
    <jointvel joint="hip_roll_r_joint" user="6"/>
    <jointvel joint="hip_pitch_r_joint" user="6"/>
    <jointvel joint="hip_yaw_r_joint" user="6"/>
    <jointvel joint="knee_pitch_r_joint" user="6"/>
    <jointvel joint="ankle_pitch_r_joint" user="6"/>
    <jointvel joint="ankle_roll_r_joint" user="6"/>
    <jointvel joint="shoulder_pitch_l_joint" user="6"/>
    <jointvel joint="shoulder_roll_l_joint" user="6"/>
    <jointvel joint="shoulder_yaw_l_joint" user="6"/>
    <jointvel joint="elbow_l_joint" user="6"/>
    <jointvel joint="shoulder_pitch_r_joint" user="6"/>
    <jointvel joint="shoulder_roll_r_joint" user="6"/>
    <jointvel joint="shoulder_yaw_r_joint" user="6"/>
    <jointvel joint="elbow_r_joint" user="6"/>

    <framequat name="orientation" objtype="site" objname="imu" noise="0.001"/>
    <framepos name="position" objtype="site" objname="imu" noise="0.001"/>
    <gyro name="angular-velocity" site="imu" noise="0.005" cutoff="34.9"/>
    <velocimeter name="linear-velocity" site="imu" noise="0.001" cutoff="30"/>
  </sensor>
</mujoco>
"""


def _iter_mjcf_bodies(elem):
    if elem.tag == "body" and "name" in elem.attrib:
        yield elem
    for child in elem:
        yield from _iter_mjcf_bodies(child)


def _load_urdf_inertials():
    urdf_root = ET.parse(URDF_PATH).getroot()
    inertials = {}
    for link in urdf_root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            continue
        inertials[link.attrib["name"]] = {
            "mass": float(mass.attrib["value"]),
            "fullinertia": (
                float(inertia.attrib["ixx"]),
                float(inertia.attrib["iyy"]),
                float(inertia.attrib["izz"]),
                float(inertia.attrib["ixy"]),
                float(inertia.attrib["ixz"]),
                float(inertia.attrib["iyz"]),
            ),
        }
    return inertials


def _load_mjcf_inertials():
    mjcf_root = ET.parse(OUTPUT_PATH).getroot()
    inertials = {}
    for body in _iter_mjcf_bodies(mjcf_root):
        inertial = body.find("inertial")
        if inertial is None:
            continue
        has_full = "fullinertia" in inertial.attrib
        has_diag = "diaginertia" in inertial.attrib
        if has_full and has_diag:
            raise RuntimeError(f"{body.attrib['name']} inertial cannot define both fullinertia and diaginertia.")
        if has_full:
            fullinertia = tuple(float(x) for x in inertial.attrib["fullinertia"].split())
        elif has_diag:
            diaginertia = [float(x) for x in inertial.attrib["diaginertia"].split()]
            fullinertia = (diaginertia[0], diaginertia[1], diaginertia[2], 0.0, 0.0, 0.0)
        else:
            continue
        inertials[body.attrib["name"]] = {
            "mass": float(inertial.attrib["mass"]),
            "fullinertia": fullinertia,
        }
    return inertials


def _validate_generated_mjcf():
    urdf_inertials = _load_urdf_inertials()
    mjcf_inertials = _load_mjcf_inertials()

    for link_name in CRITICAL_FULLINERTIA_LINKS:
        urdf_vals = urdf_inertials[link_name]["fullinertia"]
        mjcf_vals = mjcf_inertials[link_name]["fullinertia"]
        if any(abs(a - b) > INERTIA_TOLERANCE for a, b in zip(urdf_vals, mjcf_vals)):
            raise RuntimeError(
                f"{link_name} fullinertia mismatch. "
                f"Expected MuJoCo order (Ixx Iyy Izz Ixy Ixz Iyz), got {mjcf_vals}."
            )

    for link_name in CRITICAL_MASS_LINKS:
        urdf_mass = urdf_inertials[link_name]["mass"]
        mjcf_mass = mjcf_inertials[link_name]["mass"]
        if abs(urdf_mass - mjcf_mass) > MASS_TOLERANCE:
            raise RuntimeError(f"{link_name} mass mismatch: URDF={urdf_mass}, MJCF={mjcf_mass}")

    urdf_total_mass = sum(item["mass"] for item in urdf_inertials.values())
    mjcf_total_mass = sum(item["mass"] for item in mjcf_inertials.values())
    if abs(urdf_total_mass - mjcf_total_mass) > MASS_TOLERANCE:
        raise RuntimeError(
            f"Total mass mismatch: URDF={urdf_total_mass}, MJCF={mjcf_total_mass}"
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _prepare_mujoco_mesh_cache()
    OUTPUT_PATH.write_text(MJCF_TEXT, encoding="utf-8")
    _validate_generated_mjcf()
    print(f"[INFO] Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
