# Security Policy

## Supported Versions

SectionIQ is pre-1.0. Security fixes target the latest `main` branch until a
formal release policy exists.

## Reporting a Vulnerability

Please do not open public issues for vulnerabilities or accidental data exposure.
Report privately to the repository maintainer.

## Data Handling Notes

- SectionIQ stores extracted PDF text and metadata in the configured local store.
- Original PDFs are referenced by path and are not copied by the default store.
- Local stores, notebooks, spreadsheets, PDFs, logs, and benchmark outputs should
  not be committed to public repositories.
- API keys should be supplied via environment variables and never committed.
