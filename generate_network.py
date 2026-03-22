"""
generate_network.py - Generate a single 4-way intersection SUMO network.

Creates the intersection by writing SUMO XML definition files (nodes, edges)
and calling ``netconvert``.  Vehicle demand is generated with ``randomTrips.py``
from the SUMO tools.

Prerequisites
    • SUMO installed with SUMO_HOME environment variable set
    • SUMO bin/ on the system PATH

Usage
    python generate_network.py
"""
import os
import sys
import subprocess
import textwrap
import re

from config import (
    NETS_DIR, NET_FILE, ROUTE_FILE, SUMOCFG_FILE,
    APPROACH_LENGTH, NUM_LANES, SPEED_LIMIT, LEFT_HAND_TRAFFIC,
    SIMULATION_DURATION, VEHICLE_PERIOD, RANDOM_SEED,
    AMBULANCE_PERIOD, AMBULANCE_SPEED,
)


# ── Helpers ────────────────────────────────────────────────────────────

def check_sumo_home() -> str:
    """Return SUMO_HOME or exit with an informative message."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        sys.exit(
            "ERROR: SUMO_HOME environment variable is not set.\n"
            "Install SUMO from https://sumo.dlr.de/docs/Downloads.php\n"
            "then set SUMO_HOME to the installation directory."
        )
    if not os.path.isdir(sumo_home):
        sys.exit(f"ERROR: SUMO_HOME points to a non-existent path: {sumo_home}")
    return sumo_home


def _run(cmd: list[str], description: str) -> None:
    """Run a shell command, printing progress and aborting on failure."""
    print(f"\n>>> {description}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        sys.exit(f"FAILED: {description} (exit code {result.returncode})")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-5:]:
            print(f"    {line}")


# ── XML Writers ────────────────────────────────────────────────────────

def write_nodes() -> None:
    """
    Write intersection.nod.xml — five nodes forming a cross.

        north (0, L)
           |
    west (-L, 0) — center (0,0) — east (L, 0)
           |
        south (0, -L)

    The centre node is a traffic-light-controlled junction.
    Outer nodes are simple priority junctions (entry/exit points).
    """
    L = APPROACH_LENGTH
    content = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <nodes>
            <node id="center" x="0"   y="0"   type="traffic_light"/>
            <node id="north"  x="0"   y="{L}"  type="priority"/>
            <node id="south"  x="0"   y="-{L}" type="priority"/>
            <node id="east"   x="{L}"  y="0"   type="priority"/>
            <node id="west"   x="-{L}" y="0"   type="priority"/>
        </nodes>
    """)
    path = os.path.join(NETS_DIR, "intersection.nod.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Wrote: {path}")


def write_edges() -> None:
    """
    Write intersection.edg.xml — eight edges (in/out per direction).

    Each approach and departure edge has NUM_LANES lanes at SPEED_LIMIT.
    """
    content = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <edges>
            <!-- North approach -->
            <edge id="north_in"  from="north"  to="center" numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <edge id="north_out" from="center"  to="north"  numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <!-- South approach -->
            <edge id="south_in"  from="south"  to="center" numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <edge id="south_out" from="center"  to="south"  numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <!-- East approach -->
            <edge id="east_in"   from="east"   to="center" numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <edge id="east_out"  from="center"  to="east"   numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <!-- West approach -->
            <edge id="west_in"   from="west"   to="center" numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
            <edge id="west_out"  from="center"  to="west"   numLanes="{NUM_LANES}" speed="{SPEED_LIMIT}"/>
        </edges>
    """)
    path = os.path.join(NETS_DIR, "intersection.edg.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Wrote: {path}")


# ── Network Build ──────────────────────────────────────────────────────

def build_network(sumo_home: str) -> None:
    """Call ``netconvert`` to compile .nod.xml + .edg.xml → .net.xml."""
    netconvert = os.path.join(sumo_home, "bin", "netconvert")

    cmd = [
        netconvert,
        f"--node-files={os.path.join(NETS_DIR, 'intersection.nod.xml')}",
        f"--edge-files={os.path.join(NETS_DIR, 'intersection.edg.xml')}",
        f"--output-file={NET_FILE}",
        "--no-turnarounds=true",
        "--junctions.corner-detail=5",
    ]
    if LEFT_HAND_TRAFFIC:
        cmd.append("--lefthand")

    _run(cmd, "Building network with netconvert")


# ── Route Generation ──────────────────────────────────────────────────

def generate_routes(sumo_home: str, route_file: str, period: float,
                    label: str = "normal") -> None:
    """Generate vehicle demand with SUMO's ``randomTrips.py``."""
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips):
        sys.exit(f"ERROR: randomTrips.py not found at {random_trips}")

    cmd = [
        sys.executable,
        random_trips,
        f"-n={NET_FILE}",
        "-b=0",
        f"-e={SIMULATION_DURATION}",
        f"-p={period}",
        f"--route-file={route_file}",
        '--trip-attributes=departLane="best" departSpeed="max" departPos="random"',
        "--vehicle-class=passenger",
        f"--seed={RANDOM_SEED}",
        "--validate",
    ]
    _run(cmd, f"Generating routes ({label} demand, period={period}s)")


def generate_ambulance_routes(sumo_home: str, ambulance_route_file: str) -> None:
    """Generate ambulance (priority) vehicle routes with higher speed and frequency."""
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips):
        sys.exit(f"ERROR: randomTrips.py not found at {random_trips}")

    cmd = [
        sys.executable,
        random_trips,
        f"-n={NET_FILE}",
        "-b=0",
        f"-e={SIMULATION_DURATION}",
        f"-p={AMBULANCE_PERIOD}",
        f"--route-file={ambulance_route_file}",
        '--trip-attributes=departLane="best" departSpeed="max" departPos="random"',
        "--vehicle-class=emergency",
        f"--seed={RANDOM_SEED + 1}",
        "--validate",
    ]
    _run(cmd, f"Generating ambulance (priority) routes (period={AMBULANCE_PERIOD}s)")

    # Post-process ambulance routes to add speed and priority attributes
    _enhance_ambulance_routes(ambulance_route_file)


def _enhance_ambulance_routes(ambulance_route_file: str) -> None:
    """Add speed and priority attributes to ambulance vehicles in the route file."""
    if not os.path.isfile(ambulance_route_file):
        return

    with open(ambulance_route_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Add maxSpeed attribute to vehicle elements to enforce higher ambulance speed
    # Replace <vehicle id="..." type="..."> with added maxSpeed attribute
    # This regex finds vehicle tags and ensures they have maxSpeed attribute
    content = re.sub(
        r'<vehicle id="(\w+)" type="(\w+)"',
        rf'<vehicle id="\1" type="\2" maxSpeed="{AMBULANCE_SPEED:.2f}"',
        content,
    )

    with open(ambulance_route_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Enhanced ambulance routes with speed attributes")


# ── SUMO Config ───────────────────────────────────────────────────────

def write_sumocfg() -> None:
    """Write the .sumocfg that ties network + routes together."""
    content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
            <input>
                <net-file value="intersection.net.xml"/>
                <route-files value="intersection.rou.xml,intersection_ambulance.rou.xml"/>
            </input>
            <time>
                <begin value="0"/>
                <end value="3600"/>
            </time>
            <processing>
                <time-to-teleport value="-1"/>
            </processing>
        </configuration>
    """)
    with open(SUMOCFG_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n>>> Wrote SUMO config: {SUMOCFG_FILE}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(NETS_DIR, exist_ok=True)
    sumo_home = check_sumo_home()

    print("=" * 60)
    print("Network Generator — 4-Way Intersection (Indian LHT)")
    print("=" * 60)

    # 1. Write topology XML
    write_nodes()
    write_edges()

    # 2. Compile into .net.xml
    build_network(sumo_home)

    # 3. Generate vehicle demand (three densities)
    generate_routes(sumo_home, ROUTE_FILE, period=VEHICLE_PERIOD, label="normal")
    generate_routes(
        sumo_home,
        os.path.join(NETS_DIR, "intersection_low.rou.xml"),
        period=3.0,
        label="low",
    )
    generate_routes(
        sumo_home,
        os.path.join(NETS_DIR, "intersection_high.rou.xml"),
        period=0.8,
        label="high",
    )

    # 4. Generate ambulance (priority) routes
    ambulance_route_file = os.path.join(NETS_DIR, "intersection_ambulance.rou.xml")
    generate_ambulance_routes(sumo_home, ambulance_route_file)

    # 5. SUMO configuration file
    write_sumocfg()

    print("\n" + "=" * 60)
    print("Done!  Generated files in:", NETS_DIR)
    print("  - intersection.net.xml          (network)")
    print("  - intersection.rou.xml          (normal demand)")
    print("  - intersection_low.rou.xml      (low demand)")
    print("  - intersection_high.rou.xml     (high demand)")
    print("  - intersection_ambulance.rou.xml (ambulance priority vehicles)")
    print("  - intersection.sumocfg          (SUMO config)")
    print(f"\nVerify visually:  sumo-gui {SUMOCFG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
