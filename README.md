# Document Upload SDLC — Options Summary

## The Problem

When an agent uploads a procedure document to prod, there's no process to verify it's the correct, approved document. A perfectly parsed PDF with wrong procedures goes live silently. You need a way to ensure every document in prod was reviewed and approved in UAT first.

## The Core Concept

Every option below uses the same underlying mechanism: a **fingerprint** (SHA-256 hash of all chunk contents) computed when a document is approved in UAT. When the same document is uploaded in prod, the system computes the fingerprint again and checks: does it match the approved version?

The difference between options is **how the fingerprint travels from UAT to prod**, given that they're on separate AWS accounts.

---

## Option 1: Shared S3 Bucket

UAT writes a small JSON file (containing the fingerprint + approval metadata) to an S3 bucket on approval. Prod reads from the same bucket on upload.

```
UAT Account                   S3 Bucket                    Prod Account
                              (shared via IAM)
 Approve doc ──write──▶  fingerprints/doc_001.json  ◀──read── Upload doc
                              {fingerprint, approved_by,       Compare fingerprints
                               version, date}
```

| Pros | Cons |
|------|------|
| Simplest to implement — just IAM bucket policies | Not tamper-proof — anyone with S3 write access could edit the fingerprint |
| No new infrastructure, S3 already exists | Brief eventual consistency window after write |
| Built-in version history (one object per version) | Requires both accounts to access the same bucket |
| No change to agent workflow | No built-in notification when a new fingerprint is published |
| Cheapest option | |

---

## Option 2: Signed Manifest (Travels with the Document)

UAT generates a cryptographically signed JSON file (using AWS KMS asymmetric key) alongside the approved PDF. The content team gives the agent both files. Prod verifies the KMS signature to prove the manifest came from UAT, then compares fingerprints.

```
UAT Account                                                  Prod Account

 Approve doc                                                  Agent uploads
 Sign manifest  ──── PDF + manifest.json ────────────────▶   Verify KMS signature
 with KMS key         (via email, shared drive, portal)       Compare fingerprints
```

The agent can upload two files, or this can be simplified to a single file via a ZIP package, embedding the manifest in PDF metadata, or a prod upload portal that already has the manifest.

| Pros | Cons |
|------|------|
| Zero cross-account networking at runtime | Agent handles two files (unless using ZIP or embedded metadata) |
| Tamper-proof — cryptographically signed by KMS | More complex initial setup (KMS key + policy) |
| Works even if accounts are fully isolated | If parsing differs between UAT and prod, fingerprints won't match |
| Only UAT can sign — prod can only verify (key policy enforced) | If agent loses the manifest, need to re-download from portal |
| Can verify fully offline with exported public key | |

---

## Option 3: Cross-Account API

UAT calls a prod API endpoint (API Gateway + Lambda) to register the fingerprint directly in prod's registry database. No shared storage, no file passing.

```
UAT Account                                                  Prod Account

 Approve doc ────HTTPS call──▶  API Gateway ──▶ Lambda ──▶  Prod registry DB
                                 (register fingerprint)
```

| Pros | Cons |
|------|------|
| Prod has full control of its own registry | Requires network connectivity between accounts (VPC peering, PrivateLink, or public API) |
| Clean API contract with validation and rate limiting | More infrastructure to build and maintain |
| Easy to add audit logging and monitoring | If API is down when UAT approves, need retry logic |
| No files passed between teams | Most complex to set up |
| Prod never depends on external storage | Requires a platform/infra team to maintain |

---

## Option 4: Fingerprint in Code (Requires Code Deployment)

The fingerprint is stored directly in your application code (a config file, a Python constant, or a YAML checked into your repo). Every document approval requires a code change and deployment.

```
UAT Account                     Code Repository                  Prod Account

 Approve doc ──▶ PR to update ──▶ CI/CD pipeline ──▶ Deploy ──▶  App reads
                  fingerprints.yml   runs tests         to prod    fingerprints
                  in repo                                          from config
```

The config file in your repo looks like:

```yaml
# fingerprints.yml — checked into your repo
documents:
  doc_cancel_001:
    fingerprint: "a3f8c9e2d4b7..."
    version: 3
    approved_by: "john.smith"
    approved_at: "2026-04-21"
  doc_military_001:
    fingerprint: "7b2c4d9e1f6a..."
    version: 2
    approved_by: "jane.doe"
    approved_at: "2026-04-20"
```

On upload, prod reads this config and compares:

```python
import yaml

with open("fingerprints.yml") as f:
    approved = yaml.safe_load(f)

def check_fingerprint(doc_id, upload_fingerprint):
    record = approved["documents"].get(doc_id)
    if not record:
        return {"status": "no_approved_version"}
    if upload_fingerprint == record["fingerprint"]:
        return {"status": "match"}
    return {"status": "mismatch"}
```

| Pros | Cons |
|------|------|
| Simplest to understand — fingerprint is right there in the code | **Requires a code deployment for every document change** |
| Full Git history — every approval is a commit with diff and reviewer | Couples document content lifecycle to application release cycle |
| Existing CI/CD pipeline handles it — no new infrastructure | Daily document uploads mean daily deployments |
| PR review process is built in — someone must approve the code change | Engineering team becomes a bottleneck for document updates |
| No cross-account networking, no shared storage, no KMS | Agents can't upload until deployment completes |
| Tamper-proof — code changes require repo access and PR approval | Slow — document change → PR → review → merge → build → deploy → agent uploads |
| Works in any environment, no AWS-specific services needed | Doesn't scale if document volume increases |

---

## Comparison

| Factor | S3 Bucket | Signed Manifest | Cross-Account API | Code Deployment |
|--------|-----------|-----------------|-------------------|-----------------|
| **Complexity** | Low | Medium | High | Low |
| **Infrastructure needed** | S3 + IAM | KMS key + IAM | API Gateway + Lambda + DB | Git repo + CI/CD |
| **Code release per document** | No | No | No | **Yes** |
| **Cross-account networking** | S3 access | None at runtime | HTTPS connectivity | None |
| **Tamper-proof** | No | Yes (KMS signed) | Depends on API auth | Yes (PR approval) |
| **Agent workflow change** | None | Upload 2 files (or ZIP) | None | Wait for deployment |
| **Engineering involvement** | None per upload | None per upload | None per upload | **Every upload** |
| **Time from approval to live** | Minutes | Minutes | Minutes | **Hours** (deploy cycle) |
| **Audit trail** | S3 object versions | Manifest file itself | API logs | Git commit history |
| **Scales to daily uploads** | Yes | Yes | Yes | Painful |
| **Offline / no network** | No (needs S3) | Yes (manifest is a file) | No (needs API) | Yes (config in code) |

---

## Recommendation

**If daily uploads are the norm and engineering shouldn't be in the loop:** Option 1 (S3 bucket) to start. Upgrade to Option 2 (signed manifest) when tamper-proofing matters.

**If you have an infra team and want the cleanest architecture:** Option 3 (cross-account API).

**If uploads are rare (weekly/monthly) and you want maximum control with zero new infrastructure:** Option 4 (code deployment). The engineering bottleneck is acceptable when volume is low.

**If compliance requires cryptographic proof of approval and zero cross-account runtime dependencies:** Option 2 (signed manifest) with offline public key verification.
