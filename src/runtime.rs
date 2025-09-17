use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use js_sys::{WebAssembly, Function, Array, Object, Reflect, Uint8Array};
use web_sys::console;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::ast::*;
use crate::parser::MathParser;
use crate::codegen::{MathCompiler, CompiledMathFunction};

#[wasm_bindgen]
extern "C" {
    // Math functions that will be imported by generated WASM modules
    #[wasm_bindgen(js_namespace = Math)]
    fn sin(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn cos(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn tan(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn asin(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn acos(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn atan(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn log(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn exp(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn sqrt(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn abs(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn ceil(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn floor(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn round(x: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn pow(x: f64, y: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn min(x: f64, y: f64) -> f64;
    #[wasm_bindgen(js_namespace = Math)]
    fn max(x: f64, y: f64) -> f64;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledModel {
    pub id: String,
    pub name: String,
    pub expression: String,
    pub variables: Vec<String>,
    pub wasm_bytes: Vec<u8>,
    pub created_at: f64,
}

#[wasm_bindgen]
pub struct DynamicMathRuntime {
    parser: MathParser,
    compiler: MathCompiler,
    compiled_models: HashMap<String, CompiledModel>,
    loaded_instances: HashMap<String, js_sys::Function>,
}

#[wasm_bindgen]
impl DynamicMathRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DynamicMathRuntime {
        console::log_1(&"Initializing Dynamic Math Runtime".into());
        
        DynamicMathRuntime {
            parser: MathParser::new(),
            compiler: MathCompiler::new(),
            compiled_models: HashMap::new(),
            loaded_instances: HashMap::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn compile_model(
        &mut self,
        model_id: &str,
        model_name: &str,
        expression: &str,
        variables: Vec<String>
    ) -> Result<JsValue, JsValue> {
        console::log_1(&format!("Compiling model: {} - {}", model_id, expression).into());
        
        // Parse the expression
        let ast_js = self.parser.parse_expression(expression)?;
        
        // Compile to WASM
        let compiled_func = self.compiler.compile_expression_from_ast(&ast_js, variables.clone())?;
        
        // Store the compiled model
        let model = CompiledModel {
            id: model_id.to_string(),
            name: model_name.to_string(),
            expression: expression.to_string(),
            variables,
            wasm_bytes: compiled_func.wasm_bytes(),
            created_at: js_sys::Date::now(),
        };
        
        self.compiled_models.insert(model_id.to_string(), model);
        
        // Return success with model info
        serde_wasm_bindgen::to_value(&model).map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub fn load_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        let model = self.compiled_models.get(model_id)
            .ok_or_else(|| JsValue::from_str(&format!("Model {} not found", model_id)))?;
        
        console::log_1(&format!("Loading model: {}", model_id).into());
        
        // Create WASM module from bytes
        let wasm_bytes = Uint8Array::from(&model.wasm_bytes[..]);
        let module = WebAssembly::Module::new(&wasm_bytes.buffer())?;
        
        // Create imports object with math functions
        let imports = Object::new();
        let math_imports = Object::new();
        
        // Add all math function imports
        let sin_fn = Closure::wrap(Box::new(sin) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"sin".into(), sin_fn.as_ref())?;
        sin_fn.forget();
        
        let cos_fn = Closure::wrap(Box::new(cos) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"cos".into(), cos_fn.as_ref())?;
        cos_fn.forget();
        
        let tan_fn = Closure::wrap(Box::new(tan) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"tan".into(), tan_fn.as_ref())?;
        tan_fn.forget();
        
        let sqrt_fn = Closure::wrap(Box::new(sqrt) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"sqrt".into(), sqrt_fn.as_ref())?;
        sqrt_fn.forget();
        
        let pow_fn = Closure::wrap(Box::new(pow) as Box<dyn Fn(f64, f64) -> f64>);
        Reflect::set(&math_imports, &"pow".into(), pow_fn.as_ref())?;
        pow_fn.forget();
        
        let abs_fn = Closure::wrap(Box::new(abs) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"abs".into(), abs_fn.as_ref())?;
        abs_fn.forget();
        
        // Add more math functions as needed...
        let exp_fn = Closure::wrap(Box::new(exp) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"exp".into(), exp_fn.as_ref())?;
        exp_fn.forget();
        
        let log_fn = Closure::wrap(Box::new(log) as Box<dyn Fn(f64) -> f64>);
        Reflect::set(&math_imports, &"log".into(), log_fn.as_ref())?;
        log_fn.forget();
        
        Reflect::set(&imports, &"math".into(), &math_imports)?;
        
        // Instantiate the module
        let instance = WebAssembly::Instance::new(&module, &imports)?;
        
        // Get the exported calculate function
        let exports = instance.exports();
        let calculate_fn = Reflect::get(&exports, &"calculate".into())?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| JsValue::from_str("calculate function not found or not a function"))?;
        
        // Store the function for later use
        self.loaded_instances.insert(model_id.to_string(), calculate_fn);
        
        console::log_1(&format!("Successfully loaded model: {}", model_id).into());
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn execute_model(&self, model_id: &str, inputs: Vec<f64>) -> Result<f64, JsValue> {
        let model = self.compiled_models.get(model_id)
            .ok_or_else(|| JsValue::from_str(&format!("Model {} not found", model_id)))?;
        
        let calculate_fn = self.loaded_instances.get(model_id)
            .ok_or_else(|| JsValue::from_str(&format!("Model {} not loaded", model_id)))?;
        
        // Validate input count
        if inputs.len() != model.variables.len() {
            return Err(JsValue::from_str(&format!(
                "Expected {} inputs, got {}",
                model.variables.len(),
                inputs.len()
            )));
        }
        
        // Convert inputs to JavaScript values
        let js_inputs: Array = inputs.into_iter()
            .map(|x| JsValue::from_f64(x))
            .collect();
        
        // Call the function
        let result = calculate_fn.apply(&JsValue::NULL, &js_inputs)?;
        
        // Convert result back to f64
        result.as_f64()
            .ok_or_else(|| JsValue::from_str("Function did not return a number"))
    }
    
    #[wasm_bindgen]
    pub fn get_model_info(&self, model_id: &str) -> Result<JsValue, JsValue> {
        let model = self.compiled_models.get(model_id)
            .ok_or_else(|| JsValue::from_str(&format!("Model {} not found", model_id)))?;
        
        serde_wasm_bindgen::to_value(model).map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub fn list_models(&self) -> Result<JsValue, JsValue> {
        let model_list: Vec<&CompiledModel> = self.compiled_models.values().collect();
        serde_wasm_bindgen::to_value(&model_list).map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub fn remove_model(&mut self, model_id: &str) -> Result<(), JsValue> {
        self.compiled_models.remove(model_id);
        self.loaded_instances.remove(model_id);
        console::log_1(&format!("Removed model: {}", model_id).into());
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn validate_expression(&self, expression: &str) -> Result<JsValue, JsValue> {
        match self.parser.parse_expression(expression) {
            Ok(ast) => {
                let result = js_sys::Object::new();
                Reflect::set(&result, &"valid".into(), &true.into())?;
                Reflect::set(&result, &"ast".into(), &ast)?;
                Ok(result.into())
            }
            Err(e) => {
                let result = js_sys::Object::new();
                Reflect::set(&result, &"valid".into(), &false.into())?;
                Reflect::set(&result, &"error".into(), &e)?;
                Ok(result.into())
            }
        }
    }
    
    // Utility method to test a simple expression
    #[wasm_bindgen]
    pub fn test_simple_expression(&mut self, expression: &str, x: f64, y: f64) -> Result<f64, JsValue> {
        let test_id = "test_expr";
        let variables = vec!["x".to_string(), "y".to_string()];
        
        // Clean up any existing test model
        self.remove_model(test_id).ok();
        
        // Compile the test expression
        self.compile_model(test_id, "Test Expression", expression, variables)?;
        
        // Load and execute
        self.load_model(test_id)?;
        let result = self.execute_model(test_id, vec![x, y])?;
        
        // Clean up
        self.remove_model(test_id).ok();
        
        Ok(result)
    }
}
