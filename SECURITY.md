# Security and data handling

This is experimental research software. Security fixes target the current
release candidate and development branch; no support SLA is promised.

Load only checkpoints you trust and compare their published hashes before use.
Native PyTorch paths use the restricted weights-only loader. Restricted loading
reduces deserialization risk but is not a sandbox for arbitrary hostile files.
Legacy TensorFlow/HDF5 imports require separate dependencies and source provenance.
Never change a loader to unrestricted pickle merely to suppress a load error.

Dataset licensing, participant privacy and software licensing are distinct.
Use synthetic fixtures for issues and pull requests; keep credentials, raw
participant data and restricted financial sources out of repository history.
Configuration files describe research protocols, not data-access permission.

For a suspected vulnerability, use GitHub's private vulnerability-reporting
option when it is enabled. Otherwise request a private contact channel from the
maintainer without posting sensitive details in a public issue. Include the
version, affected component and a minimal synthetic reproduction when a private
channel is available. Ordinary numerical or documentation bugs can use Issues.
