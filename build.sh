#!/bin/bash

# Dynamic Mathematical Compilation Platform Build Script
echo "🧮 Building Dynamic Mathematical Compilation Platform..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/
rm -rf target/

# Build the WASM package
echo "🔧 Building WASM package..."
wasm-pack build --target web --out-dir pkg --dev

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "📁 Generated files:"
    ls -la pkg/
    echo ""
    echo "🚀 To run the demo:"
    echo "   1. Start a local HTTP server in this directory"
    echo "   2. Open http://localhost:8000/index.html in your browser"
    echo ""
    echo "💡 Quick server options:"
    echo "   - Python: python -m http.server 8000"
    echo "   - Node.js: npx http-server -p 8000"
    echo "   - VS Code: Live Server extension"
else
    echo "❌ Build failed!"
    exit 1
fi

# Optional: Run a simple HTTP server if Python is available
if command -v python3 &> /dev/null; then
    echo ""
    read -p "🤔 Start a local server now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🌐 Starting server at http://localhost:8000"
        echo "   Open this URL in your browser to try the demo!"
        python3 -m http.server 8000
    fi
fi
