#!/bin/bash

# Dynamic Mathematical Compilation Platform Build Script
echo "🧮 Building Dynamic Mathematical Compilation Platform..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo is not installed. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/
rm -rf target/

# Ensure correct file structure
if [ ! -f "src/lib.rs" ]; then
    echo "❌ src/lib.rs not found. Please run fix_structure.sh first."
    exit 1
fi

if [ ! -f "math_expression.pest" ]; then
    echo "❌ math_expression.pest not found in root directory."
    exit 1
fi

# Build the WASM package
echo "🔧 Building WASM package..."
wasm-pack build --target web --out-dir pkg

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
    echo ""
    echo "🔍 Common issues:"
    echo "   - Make sure all dependencies are in Cargo.toml"
    echo "   - Check that math_expression.pest is in the root directory"
    echo "   - Verify all Rust files are in src/ directory"
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
