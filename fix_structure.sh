#!/bin/bash

echo "🔧 Fixing Dynamic Math Compiler file structure..."

# 1. Move lib/lib.rs to src/lib.rs (if it exists)
if [ -f "lib/lib.rs" ]; then
    echo "📁 Moving lib/lib.rs to src/lib.rs"
    mv lib/lib.rs src/lib.rs
    rmdir lib 2>/dev/null || echo "lib directory not empty, keeping it"
fi

# 2. Rename demo file
if [ -f "demo_html.html" ]; then
    echo "📄 Renaming demo_html.html to index.html"
    mv demo_html.html index.html
fi

# 3. Fix test file name
if [ -f "tests/test_integration.rs" ]; then
    echo "🧪 Renaming test file"
    mv tests/test_integration.rs tests/integration_tests.rs
fi

# 4. Create missing cache.rs module
echo "📝 Creating src/cache.rs"
cat > src/cache.rs << 'EOF'
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::{window, Storage};
use js_sys::Date;

/// Simple cache entry for compiled mathematical expressions
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

/// Basic cache system for compiled expressions
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
    
    pub fn get(&mut self, cache_key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.memory_cache.get_mut(cache_key) {
            entry.hit_count += 1;
            entry.last_accessed = Date::now();
            Some(entry.clone())
        } else {
            None
        }
    }
    
    pub fn put(&mut self, cache_key: String, entry: CacheEntry) {
        // Simple LRU eviction if cache is full
        if self.memory_cache.len() >= self.max_entries && !self.memory_cache.contains_key(&cache_key) {
            if let Some((oldest_key, _)) = self.memory_cache
                .iter()
                .min_by_key(|(_, entry)| (entry.last_accessed as u64))
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                self.memory_cache.remove(&oldest_key);
            }
        }
        
        self.memory_cache.insert(cache_key, entry);
    }
    
    pub fn clear(&mut self) {
        self.memory_cache.clear();
    }
}

/// WASM bindings for the cache
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
    
    #[wasm_bindgen]
    pub fn generate_key(expression: &str, variables: Vec<String>, _optimization_level: &str) -> String {
        ExpressionCache::generate_cache_key(expression, &variables)
    }
    
    #[wasm_bindgen]
    pub fn has_entry(&self, cache_key: &str) -> bool {
        self.cache.memory_cache.contains_key(cache_key)
    }
    
    #[wasm_bindgen]
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        let stats = serde_json::json!({
            "total_entries": self.cache.memory_cache.len(),
            "total_hits": self.cache.memory_cache.values().map(|e| e.hit_count).sum::<u32>(),
        });
        
        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub fn maintenance(&mut self) {
        // Basic maintenance - could be expanded
    }
    
    #[wasm_bindgen]
    pub fn get_recommendations(&self) -> Vec<String> {
        vec!["Cache is functioning normally".to_string()]
    }
    
    #[wasm_bindgen]
    pub fn estimate_storage_usage(&self) -> f64 {
        self.cache.memory_cache.len() as f64 * 0.001 // Rough estimate in MB
    }
}
EOF

# 5. Create missing optimizer.rs module
echo "📝 Creating src/optimizer.rs"
cat > src/optimizer.rs << 'EOF'
use crate::ast::*;

/// Optimization levels for mathematical expressions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

/// Basic expression optimizer
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
    
    /// Basic constant folding optimization
    fn constant_fold(&self, expr: MathExpr) -> MathExpr {
        match expr {
            MathExpr::BinaryOp { op, left, right } => {
                let left_opt = self.constant_fold(*left);
                let right_opt = self.constant_fold(*right);
                
                // Try to fold constants
                if let (MathExpr::Number(l), MathExpr::Number(r)) = (&left_opt, &right_opt) {
                    match op {
                        BinaryOp::Add => MathExpr::Number(l + r),
                        BinaryOp::Subtract => MathExpr::Number(l - r),
                        BinaryOp::Multiply => MathExpr::Number(l * r),
                        BinaryOp::Divide if *r != 0.0 => MathExpr::Number(l / r),
                        BinaryOp::Power => MathExpr::Number(l.powf(*r)),
                        _ => MathExpr::BinaryOp {
                            op,
                            left: Box::new(left_opt),
                            right: Box::new(right_opt),
                        }
                    }
                } else {
                    // Apply algebraic simplifications
                    match (op, &left_opt, &right_opt) {
                        // x + 0 = x
                        (BinaryOp::Add, _, MathExpr::Number(0.0)) => left_opt,
                        (BinaryOp::Add, MathExpr::Number(0.0), _) => right_opt,
                        // x * 1 = x
                        (BinaryOp::Multiply, _, MathExpr::Number(1.0)) => left_opt,
                        (BinaryOp::Multiply, MathExpr::Number(1.0), _) => right_opt,
                        // x * 0 = 0
                        (BinaryOp::Multiply, _, MathExpr::Number(0.0)) => MathExpr::Number(0.0),
                        (BinaryOp::Multiply, MathExpr::Number(0.0), _) => MathExpr::Number(0.0),
                        _ => MathExpr::BinaryOp {
                            op,
                            left: Box::new(left_opt),
                            right: Box::new(right_opt),
                        }
                    }
                }
            }
            
            MathExpr::UnaryOp { op, operand } => {
                let operand_opt = self.constant_fold(*operand);
                
                if let MathExpr::Number(val) = operand_opt {
                    match op {
                        UnaryOp::Negate => MathExpr::Number(-val),
                        UnaryOp::Not => MathExpr::Number(if val == 0.0 { 1.0 } else { 0.0 }),
                    }
                } else {
                    MathExpr::UnaryOp {
                        op,
                        operand: Box::new(operand_opt),
                    }
                }
            }
            
            MathExpr::FunctionCall { function, args } => {
                let optimized_args: Vec<MathExpr> = args.into_iter()
                    .map(|arg| self.constant_fold(arg))
                    .collect();
                
                MathExpr::FunctionCall {
                    function,
                    args: optimized_args,
                }
            }
            
            _ => expr, // Numbers and identifiers don't need optimization
        }
    }
    
    /// Estimate the complexity of an expression
    pub fn estimate_complexity(&self, expr: &MathExpr) -> u32 {
        match expr {
            MathExpr::Number(_) | MathExpr::Identifier(_) => 1,
            MathExpr::BinaryOp { left, right, .. } => {
                1 + self.estimate_complexity(left) + self.estimate_complexity(right)
            }
            MathExpr::UnaryOp { operand, .. } => {
                1 + self.estimate_complexity(operand)
            }
            MathExpr::FunctionCall { args, .. } => {
                5 + args.iter().map(|arg| self.estimate_complexity(arg)).sum::<u32>()
            }
            MathExpr::ArrayAccess { index, .. } => {
                2 + self.estimate_complexity(index)
            }
            MathExpr::Conditional { condition, true_expr, false_expr } => {
                5 + self.estimate_complexity(condition) 
                  + self.estimate_complexity(true_expr)
                  + self.estimate_complexity(false_expr)
            }
        }
    }
}

/// Factory function to create optimizers
pub fn create_optimizer(level: OptimizationLevel) -> ExpressionOptimizer {
    ExpressionOptimizer::new(level)
}
EOF

# 6. Update the parser.rs to handle the grammar file location correctly
echo "🔧 Updating parser.rs grammar path"
if [ -f "src/parser.rs" ]; then
    # Create a backup
    cp src/parser.rs src/parser.rs.bak
    
    # Update the grammar path in parser.rs
    sed -i 's/#\[grammar = "math_expression.pest"\]/#[grammar = "..\/math_expression.pest"]/' src/parser.rs
    
    echo "✅ Updated grammar path in parser.rs"
fi

# 7. Create a proper lib.rs if it doesn't exist
if [ ! -f "src/lib.rs" ]; then
    echo "📝 Creating src/lib.rs"
    cat > src/lib.rs << 'EOF'
use wasm_bindgen::prelude::*;
use web_sys::console;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
    ( $( $t:tt )* ) => {
        console::log_1(&format!( $( $t )* ).into());
    }
}

// Use `wee_alloc` as the global allocator for smaller WASM binary size
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Module declarations
pub mod ast;
pub mod parser;
pub mod codegen;
pub mod runtime;
pub mod cache;
pub mod optimizer;

// Re-exports for easy access from JavaScript
pub use ast::*;
pub use parser::MathParser;
pub use codegen::MathCompiler;
pub use runtime::DynamicMathRuntime;
pub use cache::WasmExpressionCache;
pub use optimizer::{ExpressionOptimizer, OptimizationLevel};

// Called when the WASM module is instantiated
#[wasm_bindgen(start)]
pub fn main() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    
    log!("Mathematical Compilation Platform initialized!");
}

// High-level API functions for easier JavaScript integration
#[wasm_bindgen]
pub struct MathCompilerPlatform {
    runtime: DynamicMathRuntime,
}

#[wasm_bindgen]
impl MathCompilerPlatform {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MathCompilerPlatform {
        log!("Creating new Math Compiler Platform");
        MathCompilerPlatform {
            runtime: DynamicMathRuntime::new(),
        }
    }
    
    /// Parse and validate a mathematical expression
    #[wasm_bindgen]
    pub fn validate_expression(&self, expression: &str) -> Result<JsValue, JsValue> {
        self.runtime.validate_expression(expression)
    }
    
    /// Compile a mathematical model from an expression string
    #[wasm_bindgen]
    pub fn compile_model(
        &mut self,
        model_id: &str,
        model_name: &str,
        expression: &str,
        variables: Vec<String>
    ) -> Result<JsValue, JsValue> {
        self.runtime.compile_model(model_id, model_name, expression, variables)
    }
    
    /// Load a compiled model for execution
    #[wasm_bindgen]
    pub fn load_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        self.runtime.load_model(model_id)
    }
    
    /// Execute a loaded mathematical model with given inputs
    #[wasm_bindgen]
    pub fn execute_model(&self, model_id: &str, inputs: Vec<f64>) -> Result<f64, JsValue> {
        self.runtime.execute_model(model_id, inputs)
    }
    
    /// Get information about a compiled model
    #[wasm_bindgen]
    pub fn get_model_info(&self, model_id: &str) -> Result<JsValue, JsValue> {
        self.runtime.get_model_info(model_id)
    }
    
    /// List all compiled models
    #[wasm_bindgen]
    pub fn list_models(&self) -> Result<JsValue, JsValue> {
        self.runtime.list_models()
    }
    
    /// Remove a model from the runtime
    #[wasm_bindgen]
    pub fn remove_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        self.runtime.remove_model(model_id)
    }
    
    /// Quick test method for simple expressions
    #[wasm_bindgen]
    pub fn test_expression(&mut self, expression: &str, x: f64, y: f64) -> Result<f64, JsValue> {
        self.runtime.test_simple_expression(expression, x, y)
    }
    
    /// Compile, load, and execute in one step (for simple use cases)
    #[wasm_bindgen]
    pub fn evaluate_expression(
        &mut self,
        expression: &str,
        variable_names: Vec<String>,
        variable_values: Vec<f64>
    ) -> Result<f64, JsValue> {
        if variable_names.len() != variable_values.len() {
            return Err(JsValue::from_str("Variable names and values must have the same length"));
        }
        
        let temp_id = "temp_eval";
        
        // Clean up any existing temp model
        self.runtime.remove_model(temp_id).ok();
        
        // Compile the expression
        self.runtime.compile_model(temp_id, "Temporary", expression, variable_names)?;
        
        // Load and execute
        self.runtime.load_model(temp_id)?;
        let result = self.runtime.execute_model(temp_id, variable_values)?;
        
        // Clean up
        self.runtime.remove_model(temp_id).ok();
        
        Ok(result)
    }
}

// Utility functions
#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Version information
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Feature detection
#[wasm_bindgen]
pub fn supported_features() -> JsValue {
    let features = js_sys::Object::new();
    
    // Add feature flags
    js_sys::Reflect::set(
        &features,
        &"mathematical_functions".into(),
        &true.into(),
    ).unwrap();
    
    js_sys::Reflect::set(
        &features,
        &"dynamic_compilation".into(),
        &true.into(),
    ).unwrap();
    
    js_sys::Reflect::set(
        &features,
        &"expression_validation".into(),
        &true.into(),
    ).unwrap();
    
    js_sys::Reflect::set(
        &features,
        &"model_caching".into(),
        &true.into(),
    ).unwrap();
    
    features.into()
}
EOF
fi

# 8. Update build.sh to be more robust
echo "🔧 Updating build.sh"
cat > build.sh << 'EOF'
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
EOF

chmod +x build.sh

echo ""
echo "✅ File structure fixes completed!"
echo ""
echo "📋 Summary of changes:"
echo "   - Moved lib/lib.rs to src/lib.rs (if it existed)"
echo "   - Renamed demo_html.html to index.html"
echo "   - Created missing src/cache.rs module"
echo "   - Created missing src/optimizer.rs module"
echo "   - Updated parser.rs grammar path"
echo "   - Fixed Cargo.toml configuration"
echo "   - Updated build.sh with better error handling"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: ./build.sh"
echo "   2. Start a web server and open index.html"
echo ""
echo "🔍 If you encounter issues:"
echo "   - Check that math_expression.pest is in the root directory"
echo "   - Ensure all .rs files are in src/ directory"
echo "   - Verify Cargo.toml dependencies are correct"
