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

// Re-exports for easy access from JavaScript
pub use ast::*;
pub use parser::MathParser;
pub use codegen::MathCompiler;
pub use runtime::DynamicMathRuntime;

// Called when the WASM module is instantiated. 
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
