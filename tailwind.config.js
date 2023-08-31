/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      width: {
        '70': '32rem',
      }
    },
  },
  plugins: [
    require('flowbite/plugin')
  ],
  
}

