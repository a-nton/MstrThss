# GDELT BigQuery Setup Guide

This guide walks you through setting up Google Cloud BigQuery to access the full GDELT historical archive (2015-present).

## Why BigQuery?

- **GDELT DOC API**: Limited to last 3 months (rolling window)
- **GDELT BigQuery**: Full archive from February 2015 to present
- **Free Tier**: 1TB queries per month (usually sufficient for research)

## Step-by-Step Setup

### 1. Create Google Cloud Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Sign in with your Google account
3. Accept terms of service
4. **Note**: Requires credit card for verification, but you won't be charged unless you exceed free tier

### 2. Create a New Project

1. Click **"Select a project"** at the top
2. Click **"New Project"**
3. Name it (e.g., "gdelt-thesis")
4. Click **"Create"**
5. Wait for project creation (takes ~30 seconds)

### 3. Enable BigQuery API

1. In the search bar, type **"BigQuery API"**
2. Click on **"BigQuery API"**
3. Click **"Enable"**
4. Wait for API to be enabled

### 4. Create Service Account (Authentication)

1. Go to **"IAM & Admin"** → **"Service Accounts"**
2. Click **"Create Service Account"**
3. Name: `gdelt-reader` (or anything you like)
4. Description: "Read-only access for GDELT BigQuery"
5. Click **"Create and Continue"**
6. Role: Select **"BigQuery User"**
7. Click **"Continue"** → **"Done"**

### 5. Generate Credentials JSON

1. Click on the service account you just created
2. Go to **"Keys"** tab
3. Click **"Add Key"** → **"Create new key"**
4. Select **"JSON"**
5. Click **"Create"**
6. A JSON file will download (e.g., `gdelt-thesis-abc123.json`)
7. **IMPORTANT**: Keep this file secure! It grants access to your project.

### 6. Move Credentials File

Move the downloaded JSON file to your project directory:

```bash
# Option 1: Move to project root (recommended)
mv ~/Downloads/gdelt-thesis-*.json /Users/metanoid/Desktop/Master\ Thesis/Code\ New/gcp-credentials.json

# Option 2: Create a dedicated folder
mkdir -p /Users/metanoid/Desktop/Master\ Thesis/Code\ New/.credentials
mv ~/Downloads/gdelt-thesis-*.json /Users/metanoid/Desktop/Master\ Thesis/Code\ New/.credentials/bigquery.json
```

### 7. Set Environment Variable

Add this to your shell configuration file:

**For bash (~/.bash_profile or ~/.bashrc):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/metanoid/Desktop/Master Thesis/Code New/gcp-credentials.json"
```

**For zsh (~/.zshrc):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/metanoid/Desktop/Master Thesis/Code New/gcp-credentials.json"
```

**Temporary (current terminal only):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/metanoid/Desktop/Master Thesis/Code New/gcp-credentials.json"
```

### 8. Update .gitignore (IMPORTANT!)

Add to your `.gitignore` to prevent committing credentials:

```
# Google Cloud credentials
gcp-credentials.json
.credentials/
*-credentials.json
```

### 9. Install BigQuery Library

```bash
cd /Users/metanoid/Desktop/Master\ Thesis/Code\ New
source .venv/bin/activate
pip install google-cloud-bigquery
```

### 10. Test Your Setup

Run this Python test:

```python
from google.cloud import bigquery
import os

# Check credentials
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
print(f"Credentials path: {creds_path}")
print(f"File exists: {os.path.exists(creds_path) if creds_path else False}")

# Try to create client
try:
    client = bigquery.Client()
    print(f"✓ Successfully connected to project: {client.project}")

    # Test query (very small, won't use much quota)
    query = """
    SELECT COUNT(*) as total
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) = '2025-01-01'
    LIMIT 1
    """
    result = client.query(query).result()
    for row in result:
        print(f"✓ Test query successful! Found {row.total} records on 2025-01-01")

except Exception as e:
    print(f"❌ Error: {e}")
```

## Using the Plugin

Once setup is complete, the **GDELT BigQuery** plugin will appear in your launcher menu:

```
Available DATA SOURCE Plugins:
1. GDELT News API
   US English news from GDELT project (up to 200 articles/day)
2. Google News RSS
   Google News RSS feed with approximate date filtering
3. GDELT BigQuery (Historical)
   Full GDELT archive via BigQuery (2015-present, requires GCP credentials)
```

## Cost Management

### Free Tier
- **Queries**: 1TB processed per month (free)
- **Storage**: 10GB per month (free)
- **Typical GDELT query**: 10-100GB depending on date range

### Estimate Costs
- Small date range (1 month): ~10-50GB
- Medium date range (6 months): ~100-300GB
- Large date range (1 year): ~500GB-1TB

### Monitor Usage
1. Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
2. View **"Query history"** to see bytes processed
3. Check **"Billing"** → **"Reports"** for usage tracking

## Troubleshooting

### "Application Default Credentials not found"
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` is set
- Restart your terminal after setting the environment variable
- Verify the JSON file path is correct

### "Permission denied"
- Ensure service account has **"BigQuery User"** role
- Check that the correct project is selected

### "google-cloud-bigquery not installed"
- Run: `pip install google-cloud-bigquery`
- Ensure you're in the virtual environment

### "Quota exceeded"
- You've used more than 1TB this month
- Wait until next month or upgrade to paid tier
- Use smaller date ranges

## Alternative: GDELT Tables Available

The plugin queries the `gdelt-bq.gdeltv2.gkg_partitioned` table, but GDELT offers multiple tables:

- **gkg_partitioned**: Global Knowledge Graph (articles, themes, entities)
- **events_partitioned**: Coded events
- **mentions_partitioned**: Event mentions in articles

You can modify the plugin to query different tables for different use cases.

## Security Best Practices

1. **Never commit credentials** to Git
2. **Use read-only roles** (BigQuery User, not Admin)
3. **Rotate keys periodically** (create new, delete old)
4. **Monitor usage** to detect unauthorized access
5. **Use project-specific service accounts** (not your personal account)

## Need Help?

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [GDELT BigQuery Guide](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
- [Google Cloud Free Tier](https://cloud.google.com/free)
