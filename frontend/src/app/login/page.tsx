'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
    const router = useRouter()
    const [formData, setFormData] = useState({
        client_id: '',
        client_secret: '',
        sharing_key: ''
    })
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        setError('')

        try {
            // Call backend login endpoint which will proxy to Eka API
            const response = await fetch('http://localhost:8000/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    client_id: formData.client_id,
                    client_secret: formData.client_secret,
                    sharing_key: formData.sharing_key || undefined
                })
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.error || 'Login failed')
            }

            // Store tokens
            if (data.access_token) {
                localStorage.setItem('eka_access_token', data.access_token)
                localStorage.setItem('eka_refresh_token', data.refresh_token || '')
                localStorage.setItem('token_expires_in', data.expires_in?.toString() || '')
                localStorage.setItem('refresh_expires_in', data.refresh_expires_in?.toString() || '')
                
                // Redirect to dashboard
                router.push('/')
            }
        } catch (err: any) {
            setError(err.message || 'Failed to authenticate. Please check your credentials.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center p-4">
            <div className="max-w-md w-full">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-black text-gray-900 mb-2 tracking-tight">
                        TRISENSE AI
                    </h1>
                    <p className="text-sm text-gray-500 font-medium">
                        Eka API Authentication
                    </p>
                </div>

                {/* Login Card */}
                <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
                    <div className="mb-6">
                        <h2 className="text-2xl font-bold text-gray-900 mb-2">Sign In</h2>
                        <p className="text-sm text-gray-500">
                            Enter your Eka API credentials to access the dashboard
                        </p>
                    </div>

                    {error && (
                        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                            <p className="text-sm text-red-600">{error}</p>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                        {/* Demo Mode Button */}
                        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <p className="text-xs text-yellow-800 mb-2 font-semibold">🧪 Demo Mode Available</p>
                            <button
                                type="button"
                                onClick={() => {
                                    setFormData({
                                        client_id: 'demo',
                                        client_secret: 'demo',
                                        sharing_key: ''
                                    })
                                }}
                                className="text-xs text-yellow-700 hover:text-yellow-900 underline font-medium"
                            >
                                Click here to use demo credentials (for testing)
                            </button>
                        </div>

                        <div>
                            <label htmlFor="client_id" className="block text-sm font-semibold text-gray-700 mb-2">
                                Client ID <span className="text-red-500">*</span>
                            </label>
                            <input
                                id="client_id"
                                type="text"
                                required
                                value={formData.client_id}
                                onChange={(e) => setFormData({ ...formData, client_id: e.target.value })}
                                className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                placeholder="Enter your Client ID or 'demo' for testing"
                            />
                        </div>

                        <div>
                            <label htmlFor="client_secret" className="block text-sm font-semibold text-gray-700 mb-2">
                                Client Secret <span className="text-red-500">*</span>
                            </label>
                            <input
                                id="client_secret"
                                type="password"
                                required
                                value={formData.client_secret}
                                onChange={(e) => setFormData({ ...formData, client_secret: e.target.value })}
                                className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                placeholder="Enter your Client Secret or 'demo' for testing"
                            />
                        </div>

                        <div>
                            <label htmlFor="sharing_key" className="block text-sm font-semibold text-gray-700 mb-2">
                                Sharing Key <span className="text-gray-400 text-xs">(Optional)</span>
                            </label>
                            <input
                                id="sharing_key"
                                type="text"
                                value={formData.sharing_key}
                                onChange={(e) => setFormData({ ...formData, sharing_key: e.target.value })}
                                className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                                placeholder="Enter Sharing Key (if accessing another workspace)"
                            />
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                        >
                            {loading ? 'Authenticating...' : 'Sign In'}
                        </button>
                    </form>

                    <div className="mt-6 pt-6 border-t border-gray-200">
                        <p className="text-xs text-gray-500 text-center">
                            Don't have credentials?{' '}
                            <a
                                href="https://developer.eka.care"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:underline font-semibold"
                            >
                                Get API credentials from Eka Developer Console
                            </a>
                        </p>
                    </div>
                </div>

                {/* Info Box */}
                <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
                    <h3 className="text-sm font-bold text-blue-900 mb-2">How to get credentials:</h3>
                    <ol className="text-xs text-blue-800 space-y-1 list-decimal list-inside">
                        <li>Visit the Eka Developer Console</li>
                        <li>Click "Manage API Credentials"</li>
                        <li>Create a new API client</li>
                        <li>Copy your Client ID and Client Secret</li>
                    </ol>
                </div>
            </div>
        </div>
    )
}
