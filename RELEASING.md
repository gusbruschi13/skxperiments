# Releasing skxperiments

How to cut a release and publish to PyPI.

## Versions

- Pre-release (developmental): `X.Y.Z.devN` (e.g. `0.1.0.dev0`). Hidden from
  `pip install` unless `--pre`. Published manually so far.
- Final: `X.Y.Z` (e.g. `0.1.0`). Published automatically via GitHub Actions
  (trusted publishing) when a GitHub Release is published.

PyPI versions are immutable: a number cannot be reused or overwritten. Get it
right before uploading.

## One-time setup (already done)

- `.github/workflows/publish.yml`: builds and publishes on a published
  GitHub Release, via OIDC trusted publishing (no stored token).
- PyPI trusted publisher must be configured: pypi.org -> project
  skxperiments -> Manage -> Publishing -> add GitHub publisher
  (owner `gusbruschi13`, repo `skxperiments`, workflow `publish.yml`,
  environment empty).

## Corporate network note (TLS interception)

Behind the corporate proxy, TLS is intercepted and Python may reject the
self-signed root CA. Fixes for local uploads:

- `pip install pip-system-certs` (uses the Windows certificate store), or
- set `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` to the corporate CA `.pem`
  (the path may appear in `pip config list`), or
- upload from a network without TLS inspection.

The GitHub Actions runner has no interception, so automated publishing is
unaffected.

## Final release (recommended path: GitHub Actions)

1. Bump `version` in `pyproject.toml` to the final number (e.g. `0.1.0`).
   Commit and push (the library CI runs; make sure it is green).
2. Tag it: `git tag -a vX.Y.Z -m "skxperiments X.Y.Z"` and
   `git push origin vX.Y.Z`.
3. Create a GitHub Release pointing to that tag and publish it. The
   `publish.yml` workflow builds and uploads to PyPI automatically.
4. Verify: `pip install skxperiments` then
   `python -c "import skxperiments; print(skxperiments.__version__)"`.

Do not publish a GitHub Release for a version that is already on PyPI; the
upload would fail.

## Manual upload (fallback)

1. `python -m pip install --upgrade build twine`
2. `python -m build`
3. `python -m twine check dist/*`
4. (Optional) TestPyPI dry run:
   `python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
5. `python -m twine upload dist/*` (username `__token__`, password the API
   token). In PowerShell, quote the values:
   `$env:TWINE_USERNAME = "__token__"`.

## After publishing

- Revoke any API token that was exposed; prefer a project-scoped token.
- Remove local `dist/` and `build/` (already gitignored).