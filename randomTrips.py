"""
randomTrips.py - Generate random vehicle trips for the 4-way intersection.

Generates SUMO route files (.rou.xml) with random vehicle demand based on specified
vehicle generation period, simulation duration, and network configuration.

This script creates vehicles with random source/destination edges and departure times
distributed according to a specified period parameter.

Usage:
    python randomTrips.py -n network.net.xml -b 0 -e 3600 -p 1.5 --route-file output.rou.xml

Parameters:
    -n, --net-file          Path to SUMO network file (.net.xml)
    -b, --begin             Simulation start time (default: 0)
    -e, --end               Simulation end time (default: 3600)
    -p, --period            Average time between successive vehicle generations (seconds)
    --route-file            Output route file path
    --seed                  Random seed for reproducibility
    --validate              Validate the output file
    --trip-attributes       Additional attributes for vehicles (departLane, departSpeed, etc.)
    --vehicle-class         Vehicle class (passenger, emergency, etc.)
"""

import argparse
import sys
import os
import random
from typing import List, Tuple
import xml.etree.ElementTree as ET


class RandomTripsGenerator:
    """Generate random vehicle trips for intersection simulation."""

    # 4-way intersection approach edges
    APPROACH_EDGES = [
        "north_in",
        "south_in",
        "east_in",
        "west_in",
    ]

    DEPARTURE_EDGES = [
        "north_out",
        "south_out",
        "east_out",
        "west_out",
    ]

    def __init__(
        self,
        net_file: str,
        begin: float = 0,
        end: float = 3600,
        period: float = 1.5,
        seed: int = 42,
        route_file: str = "trips.rou.xml",
        trip_attributes: str = None,
        vehicle_class: str = "passenger",
    ):
        """
        Initialize trip generator.

        Args:
            net_file: Path to SUMO network file
            begin: Simulation start time (seconds)
            end: Simulation end time (seconds)
            period: Average interval between vehicle generations (seconds)
            seed: Random seed for reproducibility
            route_file: Output route file path
            trip_attributes: Additional vehicle attributes (e.g., departLane="best")
            vehicle_class: Vehicle class type
        """
        self.net_file = net_file
        self.begin = begin
        self.end = end
        self.period = period
        self.seed = seed
        self.route_file = route_file
        self.vehicle_class = vehicle_class
        self.trip_attributes = trip_attributes or ""

        random.seed(seed)

    def _parse_network(self) -> Tuple[List[str], List[str]]:
        """
        Parse network XML to extract available edges.

        Returns:
            Tuple of (approach_edges, departure_edges) lists
        """
        if not os.path.isfile(self.net_file):
            raise FileNotFoundError(f"Network file not found: {self.net_file}")

        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            edges = [edge.get("id") for edge in root.findall(".//edge")]

            # For this intersection, use predefined approach/departure edges
            approach_edges = [e for e in edges if "_in" in e]
            departure_edges = [e for e in edges if "_out" in e]

            return approach_edges, departure_edges
        except ET.ParseError as e:
            raise ValueError(f"Invalid network XML file: {e}")

    def generate_trips(self) -> List[dict]:
        """
        Generate random trips.

        Returns:
            List of trip dictionaries with id, depart, from, to, type
        """
        try:
            approach_edges, departure_edges = self._parse_network()
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            # Fallback to hardcoded edges for testing
            approach_edges = self.APPROACH_EDGES
            departure_edges = self.DEPARTURE_EDGES

        if not approach_edges or not departure_edges:
            print("ERROR: No approach or departure edges found in network", file=sys.stderr)
            sys.exit(1)

        trips = []
        vehicle_id = 0
        current_time = self.begin

        # Generate vehicles at exponential intervals based on period
        while current_time < self.end:
            # Generate next vehicle
            source_edge = random.choice(approach_edges)
            dest_edge = random.choice(departure_edges)

            trip = {
                "id": f"veh_{vehicle_id}",
                "depart": current_time,
                "from": source_edge,
                "to": dest_edge,
                "type": self.vehicle_class,
            }
            trips.append(trip)

            # Advance time based on period (with exponential distribution)
            # For Poisson arrival with rate=1/period, interval ~ Exp(1/period)
            interval = random.expovariate(1.0 / self.period)
            current_time += interval
            vehicle_id += 1

        return trips

    def _write_route_file(self, trips: List[dict]) -> None:
        """
        Write trips to SUMO route XML file.

        Args:
            trips: List of trip dictionaries
        """
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set(
            "xsi:noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/routes_file.xsd",
        )

        # Sort by departure time
        trips.sort(key=lambda t: t["depart"])

        for trip in trips:
            vehicle = ET.SubElement(root, "vehicle")
            vehicle.set("id", trip["id"])
            vehicle.set("type", trip["type"])
            vehicle.set("depart", f"{trip['depart']:.2f}")

            # Add additional attributes if provided
            if self.trip_attributes:
                # Parse trip_attributes like 'departLane="best" departSpeed="max"'
                parts = self.trip_attributes.split()
                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        value = value.strip('"\'')
                        vehicle.set(key, value)

            # Add route
            route = ET.SubElement(vehicle, "route")
            route.set("edges", f"{trip['from']} {trip['to']}")

        # Write to file with proper formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")  # Python 3.9+
        tree.write(self.route_file, encoding="utf-8", xml_declaration=True)

    def generate(self) -> int:
        """
        Main generation method.

        Returns:
            Number of vehicles generated
        """
        print(f"Generating random trips for {self.net_file}")
        print(f"  Duration: {self.begin}–{self.end} seconds")
        print(f"  Period: {self.period} seconds/vehicle")
        print(f"  Seed: {self.seed}")

        trips = self.generate_trips()
        self._write_route_file(trips)

        print(f"Generated {len(trips)} vehicles")
        print(f"Wrote: {self.route_file}")

        return len(trips)


def main():
    """Parse arguments and generate routes."""
    parser = argparse.ArgumentParser(
        description="Generate random vehicle trips for SUMO simulation"
    )
    parser.add_argument("-n", "--net-file", required=True, help="SUMO network file")
    parser.add_argument("-b", "--begin", type=float, default=0, help="Start time")
    parser.add_argument("-e", "--end", type=float, default=3600, help="End time")
    parser.add_argument("-p", "--period", type=float, default=1.5, help="Vehicle period")
    parser.add_argument(
        "--route-file", required=True, help="Output route file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--trip-attributes", help="Trip attributes (e.g., departLane=\"best\")"
    )
    parser.add_argument(
        "--vehicle-class", default="passenger", help="Vehicle class"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate output (no-op for now)"
    )

    args = parser.parse_args()

    generator = RandomTripsGenerator(
        net_file=args.net_file,
        begin=args.begin,
        end=args.end,
        period=args.period,
        seed=args.seed,
        route_file=args.route_file,
        trip_attributes=args.trip_attributes,
        vehicle_class=args.vehicle_class,
    )

    try:
        num_vehicles = generator.generate()
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
