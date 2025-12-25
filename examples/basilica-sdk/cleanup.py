# /// script
# dependencies = ["basilica-sdk>=0.10.0"]
# ///
"""List and delete Basilica deployments."""

import argparse
import os
import sys

from basilica import BasilicaClient


def main():
    parser = argparse.ArgumentParser(description="Cleanup Basilica deployments")
    parser.add_argument("--delete", metavar="NAME", help="Delete deployment by name")
    parser.add_argument("--delete-all", action="store_true", help="Delete all deployments")
    args = parser.parse_args()

    if not os.environ.get("BASILICA_API_TOKEN"):
        sys.exit("Error: BASILICA_API_TOKEN not set")

    client = BasilicaClient()
    deployments = list(client.list_deployments().deployments)

    if args.delete:
        client.delete_deployment(args.delete)
        print(f"Deleted: {args.delete}")
    elif args.delete_all:
        if not deployments:
            print("No deployments to delete.")
            return
        for dep in deployments:
            print(f"  - {dep.instance_name}")
        if input("\nDelete all? (yes/no): ").lower() == "yes":
            for dep in deployments:
                client.delete_deployment(dep.instance_name)
                print(f"Deleted: {dep.instance_name}")
    else:
        if not deployments:
            print("No active deployments.")
            return
        print(f"{'Name':<40} {'State':<10}")
        print("-" * 50)
        for dep in deployments:
            print(f"{dep.instance_name:<40} {dep.state:<10}")


if __name__ == "__main__":
    main()
