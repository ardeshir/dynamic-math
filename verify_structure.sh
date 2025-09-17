#!/bin/bash

echo "🔍 Verifying Dynamic Math Compiler file structure..."

# Required files check
required_files=(
    "Cargo.toml"
    "math_expression.pest"
    "index.html"
    "integration_example.js"
    "build.sh"
    "src/lib.rs"
    "src/ast.rs"
    "src/parser.rs"
    "src/codegen.rs"
    "src/runtime.rs"
    "src/cache.rs"
    "src/optimizer.rs"
    "tests/integration_tests.rs"
)

missing_files=()
found_files=()

echo ""
echo "📁 Checking required files:"

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
        found_files+=("$file")
    else
        echo "❌ $file (MISSING)"
        missing_files+=("$file")
    fi
done

echo ""
echo "📊 Summary:"
echo "   Found: ${#found_files[@]} files"
echo "   Missing: ${#missing_files[@]} files"

if [ ${#missing_files[@]} -eq 0 ]; then
    echo ""
    echo "🎉 All required files are present!"
    
    # Check for incorrect file structure
    echo ""
    echo "🔍 Checking for common structure issues:"
    
    if [ -d "lib" ]; then
        echo "⚠️  lib/ directory still exists - contents should be in src/"
    fi
    
    if [ -f "demo_html.html" ]; then
        echo "⚠️  demo_html.html should be renamed to index.html"
    fi
    
    if [ -f "tests/test_integration.rs" ]; then
        echo "⚠️  tests/test_integration.rs should be renamed to tests/integration_tests.rs"
    fi
    
    # Check Cargo.toml configuration
    if grep -q 'path = "src/lib.rs"' Cargo.toml; then
        echo "✅ Cargo.toml [lib] path is correct"
    else
        echo "⚠️  Cargo.toml [lib] section should include: path = \"src/lib.rs\""
    fi
    
    # Check grammar file reference in parser.rs
    if grep -q '#\[grammar = "\.\./math_expression\.pest"\]' src/parser.rs; then
        echo "✅ Grammar path in parser.rs is correct"
    elif grep -q '#\[grammar = "math_expression\.pest"\]' src/parser.rs; then
        echo "⚠️  Grammar path in parser.rs needs to be: ../math_expression.pest"
    fi
    
    echo ""
    echo "🚀 Ready to build! Run: ./build.sh"
    
else
    echo ""
    echo "❌ Missing files detected. Please ensure all required files are present:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    
    echo ""
    echo "💡 To fix missing files:"
    echo "   1. Run fix_structure.sh to create missing modules"
    echo "   2. Copy missing files from the artifacts/examples"
    echo "   3. Run this verification script again"
fi

# Check if build tools are available
echo ""
echo "🛠️  Checking build tools:"

if command -v rustc &> /dev/null; then
    rust_version=$(rustc --version)
    echo "✅ Rust: $rust_version"
else
    echo "❌ Rust not installed - run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

if command -v wasm-pack &> /dev/null; then
    wasm_pack_version=$(wasm-pack --version)
    echo "✅ wasm-pack: $wasm_pack_version"
else
    echo "❌ wasm-pack not installed - run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
fi

if command -v python3 &> /dev/null || command -v node &> /dev/null; then
    echo "✅ Web server available (python3 or node)"
else
    echo "⚠️  No web server detected - install Python 3 or Node.js for local testing"
fi

echo ""
echo "📋 Next Steps:"
echo "   1. Fix any missing files or structure issues above"
echo "   2. Run: ./build.sh"
echo "   3. Start web server: python -m http.server 8000"
echo "   4. Open: http://localhost:8000/index.html"
