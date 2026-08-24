# Attribution and modifications

VelaScene is a standalone derivative based on
[kevinchiu19/SLARM](https://github.com/kevinchiu19/SLARM), the official implementation
of *SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes*.
The source import was taken from upstream revision
[`ca6a6bc`](https://github.com/kevinchiu19/SLARM/commit/ca6a6bc815a9363840412b17b4fd6f0d9dd75c32)
on August 24, 2026.

The original authorship, `LICENSE`, paper citation, and upstream repository link are
intentionally preserved. This repository does not retain the upstream Git commit
history, is not a GitHub fork, and is not presented as the official implementation.

The initial VelaScene changes, maintained by Arnav Sharma, are:

- a project-wide VelaScene identity, including renamed model classes, entry points,
  experiment namespaces, and default data paths;
- a lightweight environment, accelerator, dependency, dataset, and checkpoint preflight
  checker at `tools/check_environment.py`;
- standard-library unit tests for the checker; and
- GitHub citation metadata that uses the SLARM paper as the preferred citation.

For the research implementation, paper, and scientific results, please credit and cite
the original SLARM authors using the citation in `README.md` or `CITATION.cff`.

The VelaScene name and later modifications do not imply endorsement by the SLARM
authors. The original MIT copyright and permission notice remain in `LICENSE`.
