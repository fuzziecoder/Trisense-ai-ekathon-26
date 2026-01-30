# Login Credentials Guide

## 🧪 Demo Mode (For Testing)

You can test the application **without real Eka API credentials** using demo mode:

### Demo Credentials:
- **Client ID**: `demo`
- **Client Secret**: `demo`
- **Sharing Key**: (leave empty)

### How to Use:
1. Go to http://localhost:3000/login
2. Click the "Click here to use demo credentials" button (or manually enter `demo` / `demo`)
3. Click "Sign In"
4. You'll be redirected to the dashboard

**Note**: Demo mode only works for testing the UI and application flow. It does not connect to the real Eka API.

---

## 🔐 Real Eka API Credentials

To use the application with real Eka API integration, you need to obtain credentials from Eka:

### Step 1: Sign Up for Eka
1. Visit: https://developer.eka.care
2. Click "Sign Up" or "Get Started"
3. Choose your sign-up method (Email, Google, Apple, or Username)
4. Complete the registration process

### Step 2: Create Workspace
1. After signing up, you'll be prompted to create a workspace
2. Enter your workspace details
3. Your account and workspace will be created automatically

### Step 3: Get API Credentials
1. Log in to the Eka Developer Console: https://developer.eka.care
2. From the dashboard, click **"Manage API Credentials"** (in Quick Actions)
3. Click **"Create"** to create a new API client
4. Fill in the required details:
   - Client Name
   - Description (optional)
   - Other required fields
5. Click **"Create"**
6. **IMPORTANT**: Copy and save:
   - **Client ID** (you can see this later)
   - **Client Secret** (shown only once - save it securely!)

### Step 4: Use Credentials
1. Go to http://localhost:3000/login
2. Enter your:
   - **Client ID** (from step 3)
   - **Client Secret** (from step 3)
   - **Sharing Key** (optional - only if accessing another workspace)
3. Click "Sign In"

---

## 🔑 Sharing Key (Optional)

If you need to access another workspace:

1. Go to the Eka Hub: https://hub.eka.care
2. Click on **"API Token"**
3. Click **"Create"** to generate a sharing key
4. Provide the required information
5. Copy the sharing key
6. Use it in the login form (optional field)

---

## 📝 Long-Lived Access Token (Alternative)

For backend-to-backend communication, you can use a Long-Lived Token:

1. Go to Eka Developer Console
2. Click "Manage API Credentials"
3. Click the three-dot menu (More options)
4. Click "Create Long Live Token"
5. Copy the token (shown only once)
6. Use it directly in API requests (Authorization header)

---

## ⚠️ Important Notes

- **Client Secret** is shown only once during creation - save it securely!
- **Long-Lived Tokens** are also shown only once - save them securely!
- Demo mode is for testing only and does not provide real API access
- Real credentials are required for production use
- Keep your credentials secure and never commit them to version control

---

## 🆘 Troubleshooting

### "Authentication failed" error:
- Verify your Client ID and Client Secret are correct
- Make sure you copied the credentials exactly (no extra spaces)
- Check that your Eka account is active
- Try creating new credentials if the old ones don't work

### "Failed to connect to Eka API" error:
- Check your internet connection
- Verify the Eka API is accessible
- Check backend logs for detailed error messages

### Demo mode not working:
- Make sure you're using exactly `demo` for both Client ID and Client Secret
- Check that the backend server is running
- Verify the backend has the latest code with demo mode support

---

## 📞 Support

For issues with Eka API credentials:
- Visit: https://developer.eka.care
- Check Eka documentation
- Contact Eka support if needed

For application issues:
- Check backend logs
- Verify servers are running
- Review error messages in browser console
