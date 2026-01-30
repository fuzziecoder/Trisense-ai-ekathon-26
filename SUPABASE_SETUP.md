# Supabase Database Setup

## Current Status
The application is currently running in **demo mode** because Supabase credentials are not configured.

## Required Credentials

To connect to Supabase, you need to provide the following environment variables:

1. **SUPABASE_URL** - Your Supabase project URL
   - Format: `https://your-project-id.supabase.co`
   - Found in: Supabase Dashboard → Settings → API → Project URL

2. **SUPABASE_KEY** - Your Supabase anon/public key
   - Format: A long string starting with `eyJ...`
   - Found in: Supabase Dashboard → Settings → API → Project API keys → `anon` `public`

## How to Set Up

### Option 1: Environment Variables (Recommended)
Create a `.env` file in the `backend` directory:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
```

### Option 2: System Environment Variables
Set them in your system environment variables.

## Database Schema

The application expects the following tables in Supabase. You can run the SQL from `backend/database.py` (see `SUPABASE_SCHEMA` constant) in your Supabase SQL Editor:

### Tables Required:
1. **patients** - Patient information
2. **vitals** - Vital signs readings
3. **alerts** - Clinical alerts

### Indexes:
- `idx_vitals_patient_time` - For efficient vital sign queries
- `idx_alerts_patient` - For efficient alert queries

## Getting Supabase Credentials

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in
3. Create a new project (or use existing)
4. Go to Settings → API
5. Copy:
   - **Project URL** → Use as `SUPABASE_URL`
   - **anon public** key → Use as `SUPABASE_KEY`

## Current Behavior

Without Supabase credentials:
- The app runs in **demo mode**
- Data is stored in memory only (not persisted)
- All database operations return demo data
- No data persistence between restarts

With Supabase credentials:
- All patient data, vitals, and alerts are persisted
- Data survives server restarts
- Real-time updates via Supabase Realtime (if enabled)

## Verification

After setting up credentials, check the server logs:
- ✅ `[OK] Connected to Supabase` - Successfully connected
- ⚠️ `[WARNING] Supabase not configured` - Still in demo mode
- ⚠️ `[WARNING] Failed to connect to Supabase: ...` - Connection error (check credentials)
