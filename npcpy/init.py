"""npc-init — scaffold an npc_team/ in the current directory."""
import sys
import argparse
from npcpy.npc_compiler import initialize_npc_project


def main():
    parser = argparse.ArgumentParser(description="Initialize an NPC team project")
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("-t", "--templates", type=str, default=None)
    parser.add_argument("-ctx", "--context", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-pr", "--provider", type=str, default=None)
    parser.add_argument("-j", "--jinxes", type=str, default=None)
    args = parser.parse_args()

    jinx_groups = [g.strip() for g in args.jinxes.split(",")] if args.jinxes else None

    initialize_npc_project(
        directory=args.directory,
        templates=args.templates,
        context=args.context,
        model=args.model,
        provider=args.provider,
        include_jinx_groups=jinx_groups,
    )


if __name__ == "__main__":
    main()
