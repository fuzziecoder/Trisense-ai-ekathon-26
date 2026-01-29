/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ['./src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            colors: {
                'risk-low': '#22c55e',
                'risk-moderate': '#eab308',
                'risk-high': '#f97316',
                'risk-critical': '#ef4444'
            }
        }
    },
    plugins: []
}
