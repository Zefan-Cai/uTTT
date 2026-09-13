# Repository split and provenance

The original `Zefan-Cai/uTTT` repository was renamed to
[Zefan-Cai/uTTT-DEV](https://github.com/Zefan-Cai/uTTT-DEV). A new, independent
[Zefan-Cai/uTTT](https://github.com/Zefan-Cai/uTTT) repository was created for
the upgraded release. All enhanced code and documentation are submitted together
in a fresh root commit, with no parent commits from the original repository.
Both remain public, matching the original visibility.

| Item | Development/original repository | Upgraded repository |
|---|---|---|
| GitHub name | `Zefan-Cai/uTTT-DEV` | `Zefan-Cai/uTTT` |
| GitHub repository ID | `1316674329` (preserved) | `1368080233` (new) |
| Default branch | `main` | `main` |
| Original Git history | Preserved | Not imported; independent root commit |
| Engineering/documentation changes | Not applied by the split | Applied here |

The original source snapshot was development commit
`36167637d6fc83b918893f8867c97cc5855a3728`. This is provenance information,
not an ancestor of the new release. At migration time the original repository
had one branch, `main`, and no tags. Its history stays in `uTTT-DEV`; verified
local Git bundles also back up the state before the rename and history reset.

The new release begins with one initial commit containing every released file.
Future changes can be committed normally on top of that independent history.

## What a rename preserves, and a new repository does not copy

GitHub repository identity and its associated stars/issues/settings stay with
the renamed development repository. They are not duplicated by submitting the
files to a new repository. The new release has its own settings, workflow
run history and future issues/pull requests. This split does not enable GitHub
Pages or publish a website.

Because the old repository name is now occupied by the new repository, an old
remote URL ending in `/uTTT.git` now targets the upgraded repository. **Do not
rely on the rename redirect to reach the original repository.**

## Updating an existing checkout

If your checkout should continue to track the original development repository:

```bash
git remote set-url origin https://github.com/Zefan-Cai/uTTT-DEV.git
git remote -v
git fetch origin
```

For a clean upgraded-release checkout:

```bash
git clone https://github.com/Zefan-Cai/uTTT.git
cd uTTT
git log --oneline main
git rev-list --parents --max-parents=0 main
```

At initial publication, the log contains one commit. The root-commit command
prints its hash without any parent hashes. If you cloned the interim version
that included development commits, make a fresh clone in a different directory
and preserve any uncommitted work rather than merging the unrelated histories.

Inspect local changes before changing branches or merging. Do not force-push a
development checkout over the upgraded release merely because the old URL was
cached. The split does not create automatic synchronization between repositories.

## Compatibility boundaries

The upgrade keeps the original research model implementations, `model_type`
identifiers, experiment configurations and result figures. It improves
launch/data/checkpoint tooling and documentation. Two intentional behavior
changes deserve attention: malformed launch environments fail earlier, and
NVS checkpoint retention counts files (default three) rather than step distance.
Explicit invalid checkpoint sources now fail instead of silently starting over.

See the [changelog](../CHANGELOG.md) for the full engineering scope. No new
model-performance measurements or trained checkpoints are claimed by the split.
