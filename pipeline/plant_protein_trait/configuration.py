"""Resolve recorded local paths without silently changing the study's database."""
from pathlib import Path
from .core import read_json


def configure_resources(args):
    config_path = getattr(args, "config", None)
    resources_path = getattr(args, "resources", None)
    if config_path:
        config = read_json(config_path)
        configured = config.get("resources_manifest")
        if not configured or not config.get("environment_prefix"):
            raise ValueError("Local config lacks resources_manifest or environment_prefix")
        if resources_path and Path(resources_path).resolve() != Path(configured).resolve():
            raise ValueError("--config and --resources refer to different resource manifests")
        resources_path = configured
        env = Path(config["environment_prefix"])
        for key in ("blastp", "hmmscan"):
            if getattr(args, key) == key:
                setattr(args, key, str(env / "bin" / key))
    if not resources_path:
        if getattr(args, "use_pfam", False):
            raise ValueError("--use-pfam requires --config or --resources; alternatively supply --pfam-db")
        return
    if getattr(args, "mode", "local") != "local":
        raise ValueError("--config/--resources are local-route options")
    path = Path(resources_path)
    if not path.is_file():
        raise ValueError(f"Resources are not ready: {path}. Run 0_download_databases.py or setup with --with-databases")
    resources = read_json(path)
    args.resources = str(path.resolve())
    database = resources.get("database")
    if not database or not resources.get("blast_db") or not resources.get("interproscan"):
        raise ValueError("Resource manifest lacks database, blast_db or interproscan")
    if args.remote_database and args.remote_database != database:
        raise ValueError("Explicit database differs from the prepared resource manifest")
    args.remote_database = database
    for name in ("blast_db", "interproscan"):
        existing = getattr(args, name)
        if existing and Path(existing).resolve() != Path(resources[name]).resolve():
            raise ValueError(f"Explicit {name} differs from resource manifest")
        setattr(args, name, resources[name])
    if not Path(args.interproscan).is_file():
        raise ValueError("InterProScan path from resource manifest is missing")
    prefix = Path(args.blast_db)
    if not any(prefix.parent.glob(prefix.name + ".*")):
        raise ValueError("BLAST database files from resource manifest are missing")
    if getattr(args, "use_pfam", False):
        if not resources.get("pfam_db"):
            raise ValueError("Standalone Pfam was not prepared; use --with-pfam during database setup")
        if args.pfam_db and Path(args.pfam_db).resolve() != Path(resources["pfam_db"]).resolve():
            raise ValueError("Explicit Pfam database differs from resource manifest")
        args.pfam_db = resources["pfam_db"]
