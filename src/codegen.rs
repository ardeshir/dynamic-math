use walrus::*;
use crate::ast::*;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodegenError {
    #[error("Walrus error: {0}")]
    WalrusError(String),
    #[error("Unknown variable: {0}")]
    UnknownVariable(String),
    #[error("Unknown function: {0}")]
    UnknownFunction(String),
    #[error("Type mismatch in expression")]
    TypeMismatch,
    #[error("Array access not yet supported in codegen")]
    ArrayAccessNotSupported,
    #[error("Conditional expressions not yet supported in codegen")]
    ConditionalNotSupported,
}

pub struct WasmCodeGenerator {
    module: Module,
    locals_map: HashMap<String, LocalId>,
    function_builder: Option<FunctionBuilder>,
}

#[wasm_bindgen]
pub struct CompiledMathFunction {
    wasm_bytes: Vec<u8>,
    variable_names: Vec<String>,
}

#[wasm_bindgen]
impl CompiledMathFunction {
    #[wasm_bindgen(getter)]
    pub fn wasm_bytes(&self) -> Vec<u8> {
        self.wasm_bytes.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn variable_names(&self) -> Vec<String> {
        self.variable_names.clone()
    }
}

impl WasmCodeGenerator {
    pub fn new() -> Self {
        let mut module = Module::default();
        
        // Import mathematical functions from JavaScript
        let math_funcs = [
            ("sin", vec![ValType::F64], vec![ValType::F64]),
            ("cos", vec![ValType::F64], vec![ValType::F64]),
            ("tan", vec![ValType::F64], vec![ValType::F64]),
            ("asin", vec![ValType::F64], vec![ValType::F64]),
            ("acos", vec![ValType::F64], vec![ValType::F64]),
            ("atan", vec![ValType::F64], vec![ValType::F64]),
            ("log", vec![ValType::F64], vec![ValType::F64]),
            ("exp", vec![ValType::F64], vec![ValType::F64]),
            ("sqrt", vec![ValType::F64], vec![ValType::F64]),
            ("abs", vec![ValType::F64], vec![ValType::F64]),
            ("ceil", vec![ValType::F64], vec![ValType::F64]),
            ("floor", vec![ValType::F64], vec![ValType::F64]),
            ("pow", vec![ValType::F64, ValType::F64], vec![ValType::F64]),
            ("min", vec![ValType::F64, ValType::F64], vec![ValType::F64]),
            ("max", vec![ValType::F64, ValType::F64], vec![ValType::F64]),
        ];
        
        for (name, params, results) in math_funcs {
            let type_id = module.types.add(&params, &results);
            module.add_import_func("math", name, type_id);
        }
        
        WasmCodeGenerator {
            module,
            locals_map: HashMap::new(),
            function_builder: None,
        }
    }
    
    pub fn compile_expression(
        &mut self, 
        expr: &MathExpr,
        variables: &[String]
    ) -> Result<CompiledMathFunction, CodegenError> {
        // Reset state
        self.locals_map.clear();
        
        // Create function type (all f64 inputs and output)
        let param_types: Vec<ValType> = variables.iter().map(|_| ValType::F64).collect();
        let result_types = vec![ValType::F64];
        let func_type = self.module.types.add(&param_types, &result_types);
        
        // Create function
        let mut func = FunctionBuilder::new(&mut self.module.types, &param_types, &result_types);
        
        // Map variables to local indices
        for (i, var_name) in variables.iter().enumerate() {
            self.locals_map.insert(var_name.clone(), LocalId::from(i as u32));
        }
        
        // Generate code for the expression
        self.function_builder = Some(func);
        self.generate_expression(expr)?;
        
        // Finish the function
        let mut func = self.function_builder.take().unwrap();
        let func_id = func.finish(vec![], &mut self.module.funcs);
        
        // Export the function
        self.module.exports.add("calculate", func_id);
        
        // Generate WASM bytes
        let wasm_bytes = self.module.emit_wasm();
        
        Ok(CompiledMathFunction {
            wasm_bytes,
            variable_names: variables.to_vec(),
        })
    }
    
    fn generate_expression(&mut self, expr: &MathExpr) -> Result<(), CodegenError> {
        let func = self.function_builder.as_mut().unwrap();
        
        match expr {
            MathExpr::Number(value) => {
                func.f64_const(*value);
                Ok(())
            }
            
            MathExpr::Identifier(name) => {
                if let Some(&local_id) = self.locals_map.get(name) {
                    func.local_get(local_id);
                    Ok(())
                } else {
                    Err(CodegenError::UnknownVariable(name.clone()))
                }
            }
            
            MathExpr::BinaryOp { op, left, right } => {
                self.generate_expression(left)?;
                self.generate_expression(right)?;
                
                match op {
                    BinaryOp::Add => func.f64_add(),
                    BinaryOp::Subtract => func.f64_sub(),
                    BinaryOp::Multiply => func.f64_mul(),
                    BinaryOp::Divide => func.f64_div(),
                    BinaryOp::Power => {
                        // Call imported pow function
                        if let Some(pow_func) = self.find_import_func("pow") {
                            func.call(pow_func);
                        } else {
                            return Err(CodegenError::UnknownFunction("pow".to_string()));
                        }
                    }
                    BinaryOp::Equal => {
                        func.f64_eq();
                        // Convert boolean to f64 (1.0 for true, 0.0 for false)
                        self.bool_to_f64(func);
                    }
                    BinaryOp::NotEqual => {
                        func.f64_ne();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::LessThan => {
                        func.f64_lt();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::LessEqual => {
                        func.f64_le();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::GreaterThan => {
                        func.f64_gt();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::GreaterEqual => {
                        func.f64_ge();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::And => {
                        // Convert to boolean logic: (left != 0) && (right != 0)
                        let local1 = func.local_tee(func.add_local(ValType::F64));
                        func.f64_const(0.0);
                        func.f64_ne();
                        
                        func.local_get(local1);
                        func.f64_const(0.0);
                        func.f64_ne();
                        
                        func.i32_and();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::Or => {
                        // Convert to boolean logic: (left != 0) || (right != 0)
                        let local1 = func.local_tee(func.add_local(ValType::F64));
                        func.f64_const(0.0);
                        func.f64_ne();
                        
                        func.local_get(local1);
                        func.f64_const(0.0);
                        func.f64_ne();
                        
                        func.i32_or();
                        self.bool_to_f64(func);
                    }
                    BinaryOp::Modulo => {
                        // f64 doesn't have modulo, we'll need to implement it
                        // mod(a, b) = a - floor(a/b) * b
                        let local_a = func.local_tee(func.add_local(ValType::F64));
                        let local_b = func.local_tee(func.add_local(ValType::F64));
                        
                        func.local_get(local_a);
                        func.local_get(local_b);
                        func.f64_div();
                        if let Some(floor_func) = self.find_import_func("floor") {
                            func.call(floor_func);
                        }
                        func.local_get(local_b);
                        func.f64_mul();
                        func.local_get(local_a);
                        func.f64_sub();
                    }
                }
                Ok(())
            }
            
            MathExpr::UnaryOp { op, operand } => {
                match op {
                    UnaryOp::Negate => {
                        func.f64_const(-1.0);
                        self.generate_expression(operand)?;
                        func.f64_mul();
                    }
                    UnaryOp::Not => {
                        self.generate_expression(operand)?;
                        func.f64_const(0.0);
                        func.f64_eq();
                        self.bool_to_f64(func);
                    }
                }
                Ok(())
            }
            
            MathExpr::FunctionCall { function, args } => {
                // Generate code for arguments
                for arg in args {
                    self.generate_expression(arg)?;
                }
                
                // Call the appropriate imported function
                let func_name = match function {
                    MathFunction::Sin => "sin",
                    MathFunction::Cos => "cos",
                    MathFunction::Tan => "tan",
                    MathFunction::Asin => "asin",
                    MathFunction::Acos => "acos",
                    MathFunction::Atan => "atan",
                    MathFunction::Log => "log",
                    MathFunction::Ln => "log", // ln is log base e
                    MathFunction::Exp => "exp",
                    MathFunction::Sqrt => "sqrt",
                    MathFunction::Abs => "abs",
                    MathFunction::Ceil => "ceil",
                    MathFunction::Floor => "floor",
                    MathFunction::Round => "round",
                    MathFunction::Min => "min",
                    MathFunction::Max => "max",
                    MathFunction::Pow => "pow",
                    _ => return Err(CodegenError::UnknownFunction(format!("{:?}", function))),
                };
                
                if let Some(func_id) = self.find_import_func(func_name) {
                    func.call(func_id);
                } else {
                    return Err(CodegenError::UnknownFunction(func_name.to_string()));
                }
                
                Ok(())
            }
            
            MathExpr::ArrayAccess { .. } => {
                Err(CodegenError::ArrayAccessNotSupported)
            }
            
            MathExpr::Conditional { .. } => {
                Err(CodegenError::ConditionalNotSupported)
            }
        }
    }
    
    fn bool_to_f64(&self, func: &mut FunctionBuilder) {
        // Convert i32 boolean (0 or 1) to f64 (0.0 or 1.0)
        func.if_else(
            ValType::F64,
            |then_func| {
                then_func.f64_const(1.0);
            },
            |else_func| {
                else_func.f64_const(0.0);
            }
        );
    }
    
    fn find_import_func(&self, name: &str) -> Option<FunctionId> {
        for (import_id, import) in self.module.imports.iter() {
            if import.name == name && import.module == "math" {
                if let ImportKind::Function(func_id) = import.kind {
                    return Some(func_id);
                }
            }
        }
        None
    }
}

#[wasm_bindgen]
pub struct MathCompiler {
    generator: WasmCodeGenerator,
}

#[wasm_bindgen]
impl MathCompiler {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MathCompiler {
        MathCompiler {
            generator: WasmCodeGenerator::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn compile_expression_from_ast(
        &mut self,
        ast_js: &JsValue,
        variables: Vec<String>
    ) -> Result<CompiledMathFunction, JsValue> {
        let ast: MathExpr = serde_wasm_bindgen::from_value(ast_js.clone())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        self.generator.compile_expression(&ast, &variables)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
