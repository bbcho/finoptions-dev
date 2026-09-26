# finoptions

Project rules for agents. The universal rules are in `~/Projects/CLAUDE/AGENTS.md`, and
the Python and finance stack rules are in `~/Projects/CLAUDE/PYTHON_FINANCE.md`. This
file wins on conflict.

- **Project type: library.** Tests need unit, integration, reference validation, and
  call-site contracts, as defined in `~/Projects/CLAUDE/rules/testing.md`.
- **Port R's fOptions faithfully.** finoptions is a Python implementation of the R
  package fOptions. Match fOptions' numerical results. The object-oriented API and the
  finite-difference Greeks are intentional departures (see `README.md`).
- **Validate against fOptions.** The R scripts in `pytest/foptions_*.R` produce the
  reference values. State the tolerance in each test and cite the reference.
