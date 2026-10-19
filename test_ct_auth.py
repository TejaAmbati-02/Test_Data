"""Standalone CommerceTools auth smoke-test for the VDI.

Hardcoded on purpose so it runs with no .env / env vars. Do NOT commit this file
(the secrets are real). Run with:  python test_ct_auth.py
"""
import requests
import truststore

# Use the OS trust store so TLS works behind the corporate proxy / MITM CA.
truststore.inject_into_ssl()

CT_CLIENT_ID = "mkUEX1UR547nsGClQ_jdA2C3"
CT_CLIENT_SECRET = "8q7bhcHmI6kUZ6eppLRTOZEEiFrs3vp-"
CT_PROJECT_KEY = "easmo-dev"
CT_REGION = "europe-west1.gcp"
CT_SCOPE = f"manage_project:{CT_PROJECT_KEY} view_audit_log:{CT_PROJECT_KEY}"

AUTH_URL = f"https://auth.{CT_REGION}.commercetools.com/oauth/token"
API_BASE = f"https://api.{CT_REGION}.commercetools.com/{CT_PROJECT_KEY}"


def get_token() -> str:
    r = requests.post(
        AUTH_URL,
        auth=(CT_CLIENT_ID, CT_CLIENT_SECRET),
        data={"grant_type": "client_credentials", "scope": CT_SCOPE},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    print(f"Token OK: {body['access_token'][:8]}... (expires_in={body.get('expires_in')}s)")
    return body["access_token"]


def smoke_test_api(token: str) -> None:
    """Optional: confirm the token actually works against the API."""
    r = requests.get(
        f"{API_BASE}/product-projections",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 1},
        timeout=30,
    )
    r.raise_for_status()
    print(f"API OK: product-projections total={r.json().get('total')}")


if __name__ == "__main__":
    token = get_token()
    smoke_test_api(token)
