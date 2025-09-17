# 🔧 Dynamic Math Compiler - File Structure Fix

Your project structure has several issues that need to be resolved. Here's exactly what to do:

## 🚨 Current Issues Identified

1. **Wrong library location**: `lib/lib.rs` should be `src/lib.rs`
2. **Missing modules**: `cache.rs` and `optimizer.rs` are missing from `src/`
3. **Grammar path issue**: Parser can't find `math_expression.pest`
4. **File naming**: `demo_html.html` should be `index.html`
5. **Cargo.toml config**: Missing proper library path configuration

## 🛠️ Quick Fix (Automated)

### Step 1: Download and run the fix script
```bash
# Create the fix script
curl -o fix_structure.sh [URL_TO_SCRIPT]
chmod +x fix_structure.sh
./fix_structure.sh
```

**OR** create the script manually using the `fix_structure.sh` artifact above.

### Step 2: Verify the fix worked
```bash
# Create verification script
curl -o verify_structure.sh [URL_TO_SCRIPT]  
chmod +x verify_structure.sh
./verify_structure.sh
```

## 🔨 Manual Fix (If you prefer to do it step by step)

### 1. Fix library structure
```bash
# Move the main library file
mv lib/lib.rs src/lib.rs
rmdir lib  # if empty

# Rename HTML demo file
mv demo_html.html index.html

# Fix test file name
mv tests/test_integration.rs tests/integration_tests.rs
```

### 2. Create missing modules

**Create `src/cache.rs`:**
```rust
// Basic cache implementation for compiled expressions
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub expression_hash: String,
    pub expression: String,
    pub variables: Vec<String>,
    pub wasm_bytes: Vec<u8>,
    pub compilation_time_ms: f64,
    pub hit_count: u32,
    pub last_accessed: f64,
    pub created_at: f64,
}

pub struct ExpressionCache {
    memory_cache: HashMap<String, CacheEntry>,
    max_entries: usize,
}

impl ExpressionCache {
    pub fn new(max_entries: usize) -> Self {
        ExpressionCache {
            memory_cache: HashMap::new(),
            max_entries,
        }
    }
    
    pub fn generate_cache_key(expression: &str, variables: &[String]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        expression.hash(&mut hasher);
        variables.hash(&mut hasher);
        
        format!("expr_{:x}", hasher.finish())
    }
    
    // Add other methods as needed...
}

#[wasm_bindgen]
pub struct WasmExpressionCache {
    cache: ExpressionCache,
}

#[wasm_bindgen]
impl WasmExpressionCache {
    #[wasm_bindgen(constructor)]
    pub fn new(max_memory_entries: usize, _max_storage_size_mb: f64) -> WasmExpressionCache {
        WasmExpressionCache {
            cache: ExpressionCache::new(max_memory_entries),
        }
    }
    
    // Add WASM bindings as needed...
}
```

**Create `src/optimizer.rs`:**
```rust
use crate::ast::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

pub struct ExpressionOptimizer {
    optimization_level: OptimizationLevel,
}

impl ExpressionOptimizer {
    pub fn new(level: OptimizationLevel) -> Self {
        ExpressionOptimizer {
            optimization_level: level,
        }
    }
    
    pub fn optimize(&mut self, expr: MathExpr) -> MathExpr {
        match self.optimization_level {
            OptimizationLevel::None => expr,
            OptimizationLevel::Basic | OptimizationLevel::Aggressive => {
                self.constant_fold(expr)
            }
        }
    }
    
    fn constant_fold(&self, expr: MathExpr) -> MathExpr {
        // Implement basic constant folding
        expr
    }
}
```

### 3. Update `src/lib.rs`
Make sure it includes all modules:
```rust
pub mod ast;
pub mod parser;
pub mod codegen;
pub mod runtime;
pub mod cache;      // Add this
pub mod optimizer;  // Add this

// Re-exports
pub use cache::WasmExpressionCache;
pub use optimizer::{ExpressionOptimizer, OptimizationLevel};
```

### 4. Fix grammar path in `src/parser.rs`
Change:
```rust
#[grammar = "math_expression.pest"]
```
To:
```rust
#[grammar = "../math_expression.pest"]
```

### 5. Update `Cargo.toml`
Add this to the `[lib]` section:
```toml
[lib]
crate-type = ["cdylib"]
path = "src/lib.rs"
```

Add missing dependencies:
```toml
serde_json = "1.0"
console_error_panic_hook = { version = "0.1", optional = true }

[dependencies.web-sys]
features = [
  "console",
  "WebAssembly", 
  "WebAssemblyModule",
  "WebAssemblyInstance", 
  "WebAssemblyMemory",
  "WebAssemblyTable",
  "Function",
  "Uint8Array",
  "Storage",        # Add this
  "Window",         # Add this
]

[features]
default = ["console_error_panic_hook", "wee_alloc"]
console_error_panic_hook = ["dep:console_error_panic_hook"]
wee_alloc = ["dep:wee_alloc"]
```

## ✅ Verification

After making these changes, your file structure should look like:

```
.
├── Cargo.toml              ✅
├── README.md               ✅
├── build.sh                ✅ 
├── index.html              ✅ (renamed from demo_html.html)
├── integration_example.js  ✅
├── math_expression.pest    ✅
├── src/
│   ├── lib.rs              ✅ (moved from lib/lib.rs)
│   ├── ast.rs              ✅
│   ├── cache.rs            ✅ (NEW)
│   ├── codegen.rs          ✅
│   ├── optimizer.rs        ✅ (NEW)
│   ├── parser.rs           ✅ (grammar path fixed)
│   └── runtime.rs          ✅
└── tests/
    └── integration_tests.rs ✅ (renamed)
```

## 🚀 Build and Test

Once the structure is fixed:

```bash
# Build the project
./build.sh

# Start a web server
python -m http.server 8000

# Open in browser
# Navigate to: http://localhost:8000/index.html
```

## 🆘 If You Still Have Issues

1. **Grammar file errors**: Make sure `math_expression.pest` is in the root directory
2. **Module not found**: Check that all `.rs` files are in the `src/` directory
3. **WASM build fails**: Make sure all dependencies are in `Cargo.toml`
4. **Import errors**: Verify that `src/lib.rs` declares all modules with `pub mod`

Run the verification script to check for common issues:
```bash
./verify_structure.sh
```

This should get your project building and running! 🎉
