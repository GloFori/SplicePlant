# Optional AI Skill

The human entry point is `pipeline/README.md`. A server-connected AI can read this folder's `SKILL.md` directly from the checkout. No proprietary host is required; the AI must actually have terminal and filesystem access to the intended server.

To copy this Skill to a directory recognized by your AI host, run from `pipeline/`:

```bash
python skill/scripts/ppt.py install --destination /absolute/path/to/your-ai-skills/plant-protein-trait
```

Use the host's documented skills directory; the command does not configure the host itself. It copies this folder and records the source pipeline location. Keep the pipeline checkout available. If moved, set `PPT_PIPELINE_ROOT` to its new absolute path. Existing destination directories are not overwritten.

Check a copied installation:

```bash
python /absolute/path/to/your-ai-skills/plant-protein-trait/scripts/ppt.py locate
```

The host may require a reload/new session to discover installed Skills. Ordinary chat-file uploads do not provide server execution access. See the human README for a complete example request and manual alternatives.
