#!/usr/bin/env python3
"""
Quick test script to verify BigQuery setup.
Run this after following BIGQUERY_SETUP.md
"""

import os
import sys

def test_environment_variable():
    """Test if GOOGLE_APPLICATION_CREDENTIALS is set"""
    print("="*50)
    print("1. Testing Environment Variable")
    print("="*50)

    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        print("\nFix: Add to your shell config (~/.zshrc or ~/.bash_profile):")
        print('export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"')
        return False

    print(f"✓ GOOGLE_APPLICATION_CREDENTIALS = {creds_path}")

    if not os.path.exists(creds_path):
        print(f"❌ Credentials file not found at: {creds_path}")
        return False

    print(f"✓ Credentials file exists")
    return True

def test_library_import():
    """Test if google-cloud-bigquery is installed"""
    print("\n" + "="*50)
    print("2. Testing Library Import")
    print("="*50)

    try:
        from google.cloud import bigquery
        print("✓ google-cloud-bigquery is installed")
        return True
    except ImportError:
        print("❌ google-cloud-bigquery not installed")
        print("\nFix: Run in your virtual environment:")
        print("pip install google-cloud-bigquery")
        return False

def test_client_connection():
    """Test if we can create a BigQuery client"""
    print("\n" + "="*50)
    print("3. Testing Client Connection")
    print("="*50)

    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        print(f"✓ Successfully connected to project: {client.project}")
        return True, client
    except Exception as e:
        print(f"❌ Failed to create BigQuery client")
        print(f"Error: {e}")
        print("\nCommon fixes:")
        print("1. Ensure service account has 'BigQuery User' role")
        print("2. Verify JSON file is valid")
        print("3. Check project ID in credentials file")
        return False, None

def test_query_execution(client):
    """Test a simple query"""
    print("\n" + "="*50)
    print("4. Testing Query Execution")
    print("="*50)

    # Very small test query
    query = """
    SELECT COUNT(*) as total
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) = '2025-01-01'
    LIMIT 1
    """

    print("Running test query (this may take 10-30 seconds)...")

    try:
        query_job = client.query(query)
        results = query_job.result()

        for row in results:
            print(f"✓ Test query successful!")
            print(f"  Found {row.total:,} GDELT records on 2025-01-01")

        # Show usage
        bytes_processed = query_job.total_bytes_processed
        bytes_billed = query_job.total_bytes_billed
        print(f"\n📊 Query Statistics:")
        print(f"  Bytes processed: {bytes_processed / 1e9:.4f} GB")
        print(f"  Bytes billed: {bytes_billed / 1e9:.4f} GB")

        return True

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

def test_plugin():
    """Test the GDELT BigQuery plugin"""
    print("\n" + "="*50)
    print("5. Testing GDELT BigQuery Plugin")
    print("="*50)

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from modules.data_sources import gdelt_bigquery

        available, error = gdelt_bigquery.check_bigquery_available()

        if available:
            print("✓ Plugin is ready to use!")
            return True
        else:
            print(f"❌ Plugin not ready: {error}")
            return False

    except Exception as e:
        print(f"❌ Plugin test failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  GDELT BigQuery Setup Verification")
    print("="*60)

    # Run tests
    if not test_environment_variable():
        print("\n❌ Setup incomplete. Fix the issues above and try again.")
        return

    if not test_library_import():
        print("\n❌ Setup incomplete. Fix the issues above and try again.")
        return

    success, client = test_client_connection()
    if not success:
        print("\n❌ Setup incomplete. Fix the issues above and try again.")
        return

    if not test_query_execution(client):
        print("\n❌ Setup incomplete. Fix the issues above and try again.")
        return

    if not test_plugin():
        print("\n❌ Setup incomplete. Fix the issues above and try again.")
        return

    print("\n" + "="*60)
    print("  ✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour BigQuery setup is complete and ready to use.")
    print("The 'GDELT BigQuery (Historical)' plugin will now appear")
    print("in your launcher menu.\n")

if __name__ == "__main__":
    main()
