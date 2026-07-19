# Google Maps Integration Setup Guide

## Overview

TripAI uses Google Maps APIs to provide distance, duration, and coordinate data.
The app works WITHOUT an API key using its built-in offline database of 100+ Indian cities.
An API key enables live data for cities not in the offline database.

## Quick Start (No API Key)

The app works out of the box! The offline fallback covers:
- 120+ Indian cities with coordinates
- 80+ popular travel routes with real distances
- Haversine-based estimation for any other city pair

## Setting Up Google Maps API (Optional)

### Step 1: Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., "TripAI")
3. Enable billing (required for Maps APIs)

### Step 2: Enable Required APIs
Enable these two APIs in the API Library:
- **Distance Matrix API** — for distance and duration data
- **Geocoding API** — for coordinate lookups

### Step 3: Create an API Key
1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > API Key**
3. Restrict the key:
   - **Application restrictions**: None (or HTTP referrers if web-only)
   - **API restrictions**: Restrict to Distance Matrix API + Geocoding API

### Step 4: Set the Environment Variable

**Windows (Command Prompt):**
```cmd
set GOOGLE_MAPS_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_MAPS_API_KEY = "your_api_key_here"
```

**Linux/macOS:**
```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

**Permanent (Windows):**
```cmd
setx GOOGLE_MAPS_API_KEY "your_api_key_here"
```

### Step 5: Test the Integration
```bash
py -3 test_maps_service.py
```

## Cost Optimization

The app uses a three-tier caching strategy to minimize API costs:

| Tier | Source | Cost | Coverage |
|------|--------|------|----------|
| 1 | SQLite Cache | Free | Previously looked-up routes (30-day TTL) |
| 2 | Offline Fallback | Free | 120+ cities, 80+ routes, Haversine for rest |
| 3 | Google Maps API | ~$5/1000 calls | Any city worldwide |

### Google Maps Pricing (as of 2025)
- Distance Matrix API: $5.00 per 1,000 elements
- Geocoding API: $5.00 per 1,000 requests
- Free tier: $200/month credit (~40,000 calls)

### Tips to Reduce Costs
1. The offline fallback handles most Indian routes — API calls are rare
2. All API results are cached for 30 days
3. Monitor usage in Google Cloud Console > APIs & Services > Dashboard
4. Set billing alerts at $10/month
5. Consider restricting the API key to prevent unauthorized usage

## Architecture

```
User Request
    │
    ▼
┌─────────────────────┐
│   MapsService       │
│                     │
│  1. SQLite Cache ◄──┼── Fastest, free
│       │ miss        │
│  2. Offline Data ◄──┼── 120+ cities, free  
│       │ miss        │
│  3. Google API  ◄───┼── Live data, cached
│       │             │
│  ▼ Result           │
│  Cache & Return     │
└─────────────────────┘
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not set" | Set `GOOGLE_MAPS_API_KEY` environment variable |
| "REQUEST_DENIED" | Enable Distance Matrix & Geocoding APIs in Cloud Console |
| "OVER_QUERY_LIMIT" | Check billing, set up billing alerts |
| Distances look wrong | Clear cache: delete `data/travel.db` and restart |
| App works without key | Expected! Offline fallback covers most Indian cities |
