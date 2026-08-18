# Security policy

## Supported versions

The latest release on the `main` branch is supported.

## Reporting a vulnerability

If you find a security issue, **please do not open a public issue**. Email
**chdavidfm@gmail.com** with:

- a description of the problem and its impact,
- the steps to reproduce it,
- the affected version or commit.

You will receive an acknowledgement within 72 hours and an assessment of the
scope within 7 days.

## Design notes

- Credentials are read from the environment or a `.env` file that is **never**
  committed; it is excluded in `.gitignore`.
- The default mode runs without credentials and without network access.
- The index cache stores JSON and NumPy archives, never `pickle`, so loading a
  cache file cannot execute code.
- The container image runs the service as an unprivileged user.
- CodeQL analyses the code on every change and weekly.
