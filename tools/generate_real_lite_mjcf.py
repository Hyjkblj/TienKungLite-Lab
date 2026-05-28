from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "mjcf"
OUTPUT_PATH = OUTPUT_DIR / "real_lite.xml"
URDF_PATH = ROOT / "real_lite_lab" / "assets" / "tienkung2_lite_real" / "urdf" / "humanoid_publish.urdf"

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

# WARNING: This MJCF is hand-converted from the URDF at:
#   real_lite_lab/assets/tienkung2_lite_real/urdf/humanoid_publish.urdf
# If the URDF changes (joint ranges, meshes, inertials, collision geometry),
# this MJCF MUST be updated manually to match. sim2sim results are only
# valid when MJCF and URDF agree on joint limits and dynamics.

MJCF_TEXT = """<mujoco model="real_lite">
  <option gravity="0 0 -9.81" solver="PGS"/>
  <option integrator="implicitfast"/>
  <size njmax="500" nconmax="100"/>
  <compiler angle="radian" meshdir="../real_lite_lab/assets/tienkung2_lite_real/meshes/" eulerseq="zyx"/>

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
                  <geom name="toe1_left" contype="2" conaffinity="1" size="0.015 0.115" pos="0.035 0.025 -0.042" quat="0.707105 0 0.707108 0" type="cylinder" rgba="0.75 0.75 0.75 1"/>
                  <geom name="toe2_left" contype="2" conaffinity="1" size="0.015 0.115" pos="0.035 -0.025 -0.042" quat="0.707105 0 0.707108 0" type="cylinder" rgba="0.75 0.75 0.75 1"/>
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
                  <geom name="toe1_right" contype="2" conaffinity="1" size="0.015 0.115" pos="0.035 0.025 -0.042" quat="0.707105 0 0.707108 0" type="cylinder" rgba="0.75 0.75 0.75 1"/>
                  <geom name="toe2_right" contype="2" conaffinity="1" size="0.015 0.115" pos="0.035 -0.025 -0.042" quat="0.707105 0 0.707108 0" type="cylinder" rgba="0.75 0.75 0.75 1"/>
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
    OUTPUT_PATH.write_text(MJCF_TEXT, encoding="utf-8")
    _validate_generated_mjcf()
    print(f"[INFO] Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
