// Step-by-Step Implementation: From Simple Expressions to Virtual Compiler

// STEP 1: Understanding the Expression Journey
// ==========================================

// User inputs: "x^2 + sin(y * 3.14159)"
// Let's trace this through the entire system...

use crate::ast::*;
use crate::parser::*;
use crate::optimizer::*;
use crate::codegen::*;

/// STEP 1A: Lexical Analysis (Tokenization)
/// Input: "x^2 + sin(y * 3.14159)"
/// Output: [Identifier("x"), Power, Number(2), Plus, Function("sin"), ...]
pub struct DetailedTokenizer {
    input: String,
    position: usize,
}

impl DetailedTokenizer {
    pub fn tokenize_step_by_step(&mut self) -> Vec<Token> {
        println!("🔍 TOKENIZATION PROCESS:");
        println!("Input: {}", self.input);
        
        let mut tokens = Vec::new();
        
        while self.position < self.input.len() {
            let token = self.next_token();
            println!("  Position {}: Found token {:?}", self.position, token);
            tokens.push(token);
        }
        
        tokens
    }
    
    fn next_token(&mut self) -> Token {
        // Simplified tokenizer logic
        self.skip_whitespace();
        
        match self.current_char() {
            'x' => { self.advance(); Token::Identifier("x".to_string()) }
            '^' => { self.advance(); Token::Power }
            '+' => { self.advance(); Token::Plus }
            '(' => { self.advance(); Token::LeftParen }
            ')' => { self.advance(); Token::RightParen }
            '*' => { self.advance(); Token::Multiply }
            '0'..='9' => self.read_number(),
            'a'..='z' => self.read_identifier(),
            _ => { self.advance(); Token::Unknown }
        }
    }
    
    fn read_number(&mut self) -> Token {
        let mut number_str = String::new();
        
        while self.position < self.input.len() && 
              (self.current_char().is_ascii_digit() || self.current_char() == '.') {
            number_str.push(self.current_char());
            self.advance();
        }
        
        Token::Number(number_str.parse().unwrap_or(0.0))
    }
    
    fn read_identifier(&mut self) -> Token {
        let mut ident = String::new();
        
        while self.position < self.input.len() && 
              (self.current_char().is_alphanumeric() || self.current_char() == '_') {
            ident.push(self.current_char());  
            self.advance();
        }
        
        // Check if it's a function name
        match ident.as_str() {
            "sin" => Token::Function("sin".to_string()),
            "cos" => Token::Function("cos".to_string()),
            "sqrt" => Token::Function("sqrt".to_string()),
            _ => Token::Identifier(ident),
        }
    }
    
    fn current_char(&self) -> char {
        self.input.chars().nth(self.position).unwrap_or('\0')
    }
    
    fn advance(&mut self) {
        self.position += 1;
    }
    
    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() && self.current_char().is_whitespace() {
            self.advance();
        }
    }
}

#[derive(Debug, Clone)]
pub enum Token {
    Identifier(String),
    Function(String), 
    Number(f64),
    Plus, Minus, Multiply, Divide, Power,
    LeftParen, RightParen,
    Unknown,
}

/// STEP 1B: Syntax Analysis (Parsing to AST)
/// Transforms tokens into a tree structure respecting operator precedence
pub struct DetailedParser {
    tokens: Vec<Token>,
    position: usize,
}

impl DetailedParser {
    pub fn parse_with_explanation(&mut self) -> MathExpr {
        println!("\n🌳 PARSING PROCESS (Building AST):");
        println!("Tokens: {:?}", self.tokens);
        println!("Applying operator precedence rules...\n");
        
        let result = self.parse_expression(0); // Start with lowest precedence
        
        println!("Final AST:");
        self.print_ast(&result, 0);
        
        result
    }
    
    // Precedence climbing parser
    fn parse_expression(&mut self, min_precedence: u8) -> MathExpr {
        let mut left = self.parse_primary();
        
        while let Some(op_token) = self.peek_token() {
            let precedence = self.get_precedence(op_token);
            
            if precedence < min_precedence {
                break;
            }
            
            println!("  Parsing binary operation: {:?} (precedence: {})", op_token, precedence);
            
            let op = self.consume_operator();
            let right_precedence = if self.is_right_associative(&op) {
                precedence
            } else {
                precedence + 1
            };
            
            let right = self.parse_expression(right_precedence);
            
            left = MathExpr::BinaryOp {
                op: self.token_to_binary_op(op),
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        left
    }
    
    fn parse_primary(&mut self) -> MathExpr {
        match self.consume_token() {
            Token::Number(n) => {
                println!("    Parsed number: {}", n);
                MathExpr::Number(n)
            }
            Token::Identifier(name) => {
                println!("    Parsed identifier: {}", name);
                MathExpr::Identifier(name)
            }
            Token::Function(name) => {
                println!("    Parsing function call: {}", name);
                self.expect_token(Token::LeftParen);
                
                let mut args = Vec::new();
                if !self.check_token(&Token::RightParen) {
                    loop {
                        args.push(self.parse_expression(0));
                        if self.check_token(&Token::RightParen) {
                            break;
                        }
                        // Expect comma for multiple arguments (simplified)
                    }
                }
                
                self.expect_token(Token::RightParen);
                
                MathExpr::FunctionCall {
                    function: MathFunction::from_str(&name).unwrap(),
                    args,
                }
            }
            Token::LeftParen => {
                let expr = self.parse_expression(0);
                self.expect_token(Token::RightParen);
                expr
            }
            _ => panic!("Unexpected token in primary expression"),
        }
    }
    
    fn get_precedence(&self, token: &Token) -> u8 {
        match token {
            Token::Power => 4,        // Highest precedence
            Token::Multiply | Token::Divide => 3,
            Token::Plus | Token::Minus => 2,
            _ => 0,
        }
    }
    
    fn is_right_associative(&self, token: &Token) -> bool {
        matches!(token, Token::Power) // Power is right-associative: 2^3^4 = 2^(3^4)
    }
    
    fn print_ast(&self, expr: &MathExpr, indent: usize) {
        let spaces = "  ".repeat(indent);
        match expr {
            MathExpr::Number(n) => println!("{}Number({})", spaces, n),
            MathExpr::Identifier(name) => println!("{}Identifier({})", spaces, name),
            MathExpr::BinaryOp { op, left, right } => {
                println!("{}BinaryOp({:?})", spaces, op);
                self.print_ast(left, indent + 1);
                self.print_ast(right, indent + 1);
            }
            MathExpr::FunctionCall { function, args } => {
                println!("{}FunctionCall({:?})", spaces, function);
                for arg in args {
                    self.print_ast(arg, indent + 1);
                }
            }
            _ => println!("{}Other", spaces),
        }
    }
    
    // Helper methods (simplified)
    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }
    
    fn consume_token(&mut self) -> Token {
        let token = self.tokens[self.position].clone();
        self.position += 1;
        token
    }
    
    fn consume_operator(&mut self) -> Token {
        self.consume_token()
    }
    
    fn expect_token(&mut self, expected: Token) {
        let token = self.consume_token();
        // In real implementation, check if token matches expected
    }
    
    fn check_token(&self, expected: &Token) -> bool {
        // Simplified check
        true
    }
    
    fn token_to_binary_op(&self, token: Token) -> BinaryOp {
        match token {
            Token::Plus => BinaryOp::Add,
            Token::Minus => BinaryOp::Subtract,
            Token::Multiply => BinaryOp::Multiply,
            Token::Divide => BinaryOp::Divide,
            Token::Power => BinaryOp::Power,
            _ => panic!("Not a binary operator"),
        }
    }
}

/// STEP 2: AST Optimization Process
/// Takes the AST and applies mathematical optimizations
pub struct DetailedOptimizer {
    optimization_level: OptimizationLevel,
}

impl DetailedOptimizer {
    pub fn optimize_with_explanation(&mut self, expr: MathExpr) -> MathExpr {
        println!("\n⚡ OPTIMIZATION PROCESS:");
        println!("Original expression:");
        self.print_optimization_step(&expr, "ORIGINAL");
        
        // Step 1: Constant folding
        let folded = self.constant_fold_with_logging(expr);
        self.print_optimization_step(&folded, "CONSTANT FOLDED");
        
        // Step 2: Algebraic simplification
        let simplified = self.algebraic_simplify_with_logging(folded);
        self.print_optimization_step(&simplified, "ALGEBRAICALLY SIMPLIFIED");
        
        // Step 3: Strength reduction
        let reduced = self.strength_reduce_with_logging(simplified);
        self.print_optimization_step(&reduced, "STRENGTH REDUCED");
        
        reduced
    }
    
    fn constant_fold_with_logging(&self, expr: MathExpr) -> MathExpr {
        match expr {
            MathExpr::BinaryOp { op, left, right } => {
                let left_opt = self.constant_fold_with_logging(*left);
                let right_opt = self.constant_fold_with_logging(*right);
                
                // Check if both operands are now constants
                if let (MathExpr::Number(l), MathExpr::Number(r)) = (&left_opt, &right_opt) {
                    let result = match op {
                        BinaryOp::Add => l + r,
                        BinaryOp::Subtract => l - r,
                        BinaryOp::Multiply => l * r,
                        BinaryOp::Divide if *r != 0.0 => l / r,
                        BinaryOp::Power => l.powf(*r),
                        _ => return MathExpr::BinaryOp { op, left: Box::new(left_opt), right: Box::new(right_opt) },
                    };
                    
                    println!("  Constant folding: {} {:?} {} = {}", l, op, r, result);
                    MathExpr::Number(result)
                } else {
                    MathExpr::BinaryOp { op, left: Box::new(left_opt), right: Box::new(right_opt) }
                }
            }
            
            MathExpr::FunctionCall { function, args } => {
                let optimized_args: Vec<MathExpr> = args.into_iter()
                    .map(|arg| self.constant_fold_with_logging(arg))
                    .collect();
                
                // If all arguments are constants, evaluate the function
                if optimized_args.iter().all(|arg| matches!(arg, MathExpr::Number(_))) {
                    let constants: Vec<f64> = optimized_args.iter()
                        .map(|arg| if let MathExpr::Number(val) = arg { *val } else { 0.0 })
                        .collect();
                        
                    let result = match function {
                        MathFunction::Sin if constants.len() == 1 => constants[0].sin(),
                        MathFunction::Cos if constants.len() == 1 => constants[0].cos(),
                        MathFunction::Sqrt if constants.len() == 1 => constants[0].sqrt(),
                        _ => return MathExpr::FunctionCall { function, args: optimized_args },
                    };
                    
                    println!("  Function evaluation: {:?}({}) = {}", function, constants[0], result);
                    MathExpr::Number(result)
                } else {
                    MathExpr::FunctionCall { function, args: optimized_args }
                }
            }
            
            _ => expr,
        }
    }
    
    fn algebraic_simplify_with_logging(&self, expr: MathExpr) -> MathExpr {
        match expr {
            MathExpr::BinaryOp { op, left, right } => {
                let left_opt = self.algebraic_simplify_with_logging(*left);
                let right_opt = self.algebraic_simplify_with_logging(*right);
                
                match (op, &left_opt, &right_opt) {
                    // x + 0 = x
                    (BinaryOp::Add, _, MathExpr::Number(0.0)) => {
                        println!("  Algebraic simplification: expr + 0 = expr");
                        left_opt
                    }
                    (BinaryOp::Add, MathExpr::Number(0.0), _) => {
                        println!("  Algebraic simplification: 0 + expr = expr");
                        right_opt
                    }
                    
                    // x * 1 = x
                    (BinaryOp::Multiply, _, MathExpr::Number(1.0)) => {
                        println!("  Algebraic simplification: expr * 1 = expr");
                        left_opt
                    }
                    (BinaryOp::Multiply, MathExpr::Number(1.0), _) => {
                        println!("  Algebraic simplification: 1 * expr = expr");
                        right_opt
                    }
                    
                    // x * 0 = 0
                    (BinaryOp::Multiply, _, MathExpr::Number(0.0)) |
                    (BinaryOp::Multiply, MathExpr::Number(0.0), _) => {
                        println!("  Algebraic simplification: expr * 0 = 0");
                        MathExpr::Number(0.0)
                    }
                    
                    // x^1 = x
                    (BinaryOp::Power, _, MathExpr::Number(1.0)) => {
                        println!("  Algebraic simplification: expr^1 = expr");
                        left_opt
                    }
                    
                    // x^0 = 1
                    (BinaryOp::Power, _, MathExpr::Number(0.0)) => {
                        println!("  Algebraic simplification: expr^0 = 1");
                        MathExpr::Number(1.0)
                    }
                    
                    _ => MathExpr::BinaryOp { op, left: Box::new(left_opt), right: Box::new(right_opt) },
                }
            }
            _ => expr,
        }
    }
    
    fn strength_reduce_with_logging(&self, expr: MathExpr) -> MathExpr {
        match expr {
            MathExpr::BinaryOp { op, left, right } => {
                let left_opt = self.strength_reduce_with_logging(*left);
                let right_opt = self.strength_reduce_with_logging(*right);
                
                match (&op, &right_opt) {
                    // x^2 → x*x (multiplication is typically faster)
                    (BinaryOp::Power, MathExpr::Number(2.0)) => {
                        println!("  Strength reduction: expr^2 → expr * expr");
                        MathExpr::BinaryOp {
                            op: BinaryOp::Multiply,
                            left: Box::new(left_opt.clone()),
                            right: Box::new(left_opt),
                        }
                    }
                    
                    // x^0.5 → sqrt(x)
                    (BinaryOp::Power, MathExpr::Number(0.5)) => {
                        println!("  Strength reduction: expr^0.5 → sqrt(expr)");
                        MathExpr::FunctionCall {
                            function: MathFunction::Sqrt,
                            args: vec![left_opt],
                        }
                    }
                    
                    _ => MathExpr::BinaryOp { op, left: Box::new(left_opt), right: Box::new(right_opt) },
                }
            }
            _ => expr,
        }
    }
    
    fn print_optimization_step(&self, expr: &MathExpr, step_name: &str) {
        println!("  {}: {}", step_name, self.expr_to_string(expr));
    }
    
    fn expr_to_string(&self, expr: &MathExpr) -> String {
        match expr {
            MathExpr::Number(n) => n.to_string(),
            MathExpr::Identifier(name) => name.clone(),
            MathExpr::BinaryOp { op, left, right } => {
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Subtract => "-", 
                    BinaryOp::Multiply => "*",
                    BinaryOp::Divide => "/",
                    BinaryOp::Power => "^",
                    _ => "?",
                };
                format!("({} {} {})", self.expr_to_string(left), op_str, self.expr_to_string(right))
            }
            MathExpr::FunctionCall { function, args } => {
                let func_name = format!("{:?}", function).to_lowercase();
                let args_str = args.iter()
                    .map(|arg| self.expr_to_string(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", func_name, args_str)
            }
            _ => "complex_expr".to_string(),
        }
    }
}

/// STEP 3: WASM Code Generation
/// Converts optimized AST to WebAssembly bytecode
pub struct DetailedCodeGenerator {
    wasm_instructions: Vec<WasmInstruction>,
    local_variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum WasmInstruction {
    F64Const(f64),
    LocalGet(u32),
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    Call(String),
}

impl DetailedCodeGenerator {
    pub fn generate_with_explanation(&mut self, expr: &MathExpr, variables: &[String]) -> Vec<WasmInstruction> {
        println!("\n🏗️ WASM CODE GENERATION:");
        println!("Variables: {:?}", variables);
        self.local_variables = variables.to_vec();
        
        println!("Generating WASM instructions...");
        self.generate_expr_instructions(expr);
        
        println!("\nGenerated WASM instructions:");
        for (i, instruction) in self.wasm_instructions.iter().enumerate() {
            println!("  {}: {:?}", i, instruction);
        }
        
        println!("\nWASM Text Format:");
        self.print_wasm_text_format();
        
        self.wasm_instructions.clone()
    }
    
    fn generate_expr_instructions(&mut self, expr: &MathExpr) {
        match expr {
            MathExpr::Number(value) => {
                println!("  Generating: f64.const {}", value);
                self.wasm_instructions.push(WasmInstruction::F64Const(*value));
            }
            
            MathExpr::Identifier(name) => {
                let local_index = self.local_variables.iter()
                    .position(|var| var == name)
                    .expect("Variable not found") as u32;
                
                println!("  Generating: local.get {} ({})", local_index, name);
                self.wasm_instructions.push(WasmInstruction::LocalGet(local_index));
            }
            
            MathExpr::BinaryOp { op, left, right } => {
                println!("  Generating binary operation: {:?}", op);
                
                // Generate left operand
                self.generate_expr_instructions(left);
                
                // Generate right operand  
                self.generate_expr_instructions(right);
                
                // Generate operation
                let wasm_op = match op {
                    BinaryOp::Add => {
                        println!("  Generating: f64.add");
                        WasmInstruction::F64Add
                    }
                    BinaryOp::Subtract => {
                        println!("  Generating: f64.sub");
                        WasmInstruction::F64Sub
                    }
                    BinaryOp::Multiply => {
                        println!("  Generating: f64.mul");
                        WasmInstruction::F64Mul
                    }
                    BinaryOp::Divide => {
                        println!("  Generating: f64.div");
                        WasmInstruction::F64Div
                    }
                    BinaryOp::Power => {
                        println!("  Generating: call $pow");
                        WasmInstruction::Call("pow".to_string())
                    }
                    _ => panic!("Unsupported binary operation"),
                };
                
                self.wasm_instructions.push(wasm_op);
            }
            
            MathExpr::FunctionCall { function, args } => {
                let func_name = match function {
                    MathFunction::Sin => "sin",
                    MathFunction::Cos => "cos", 
                    MathFunction::Sqrt => "sqrt",
                    _ => "unknown_func",
                };
                
                println!("  Generating function call: {}", func_name);
                
                // Generate arguments
                for arg in args {
                    self.generate_expr_instructions(arg);
                }
                
                println!("  Generating: call ${}", func_name);
                self.wasm_instructions.push(WasmInstruction::Call(func_name.to_string()));
            }
            
            _ => println!("  Unsupported expression type"),
        }
    }
    
    fn print_wasm_text_format(&self) {
        println!("(module");
        println!("  ;; Import math functions");
        println!("  (import \"math\" \"sin\" (func $sin (param f64) (result f64)))");
        println!("  (import \"math\" \"cos\" (func $cos (param f64) (result f64)))");
        println!("  (import \"math\" \"sqrt\" (func $sqrt (param f64) (result f64)))");
        println!("  (import \"math\" \"pow\" (func $pow (param f64 f64) (result f64)))");
        println!();
        
        // Generate function signature
        print!("  (func $calculate");
        for (i, var) in self.local_variables.iter().enumerate() {
            print!(" (param ${} f64)", var);
        }
        println!(" (result f64)");
        
        // Generate function body
        for instruction in &self.wasm_instructions {
            let instr_str = match instruction {
                WasmInstruction::F64Const(val) => format!("    f64.const {}", val),
                WasmInstruction::LocalGet(idx) => format!("    local.get {}", idx),
                WasmInstruction::F64Add => "    f64.add".to_string(),
                WasmInstruction::F64Sub => "    f64.sub".to_string(),
                WasmInstruction::F64Mul => "    f64.mul".to_string(),
                WasmInstruction::F64Div => "    f64.div".to_string(),
                WasmInstruction::Call(func) => format!("    call ${}", func),
            };
            println!("{}", instr_str);
        }
        
        println!("  )");
        println!("  (export \"calculate\" (func $calculate))");
        println!(")");
    }
}

/// STEP 4: Runtime Execution Example
/// Shows how the compiled WASM function gets executed
pub struct DetailedRuntime {
    compiled_functions: std::collections::HashMap<String, Vec<WasmInstruction>>,
}

impl DetailedRuntime {
    pub fn execute_with_explanation(&self, function_name: &str, inputs: &[f64]) -> f64 {
        println!("\n🚀 RUNTIME EXECUTION:");
        println!("Function: {}", function_name);
        println!("Inputs: {:?}", inputs);
        
        let instructions = self.compiled_functions.get(function_name)
            .expect("Function not found");
        
        println!("\nSimulating WASM stack machine execution:");
        
        // Simulate WASM stack machine
        let mut stack: Vec<f64> = Vec::new();
        let mut locals: Vec<f64> = inputs.to_vec();
        
        println!("Initial locals: {:?}", locals);
        println!("Initial stack: {:?}", stack);
        println!();
        
        for (pc, instruction) in instructions.iter().enumerate() {
            print!("Step {}: {:?} -> ", pc, instruction);
            
            match instruction {
                WasmInstruction::F64Const(val) => {
                    stack.push(*val);
                    println!("Stack: {:?}", stack);
                }
                
                WasmInstruction::LocalGet(idx) => {
                    let val = locals[*idx as usize];
                    stack.push(val);
                    println!("Pushed local[{}] = {} -> Stack: {:?}", idx, val, stack);
                }
                
                WasmInstruction::F64Add => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    let result = a + b;
                    stack.push(result);
                    println!("{} + {} = {} -> Stack: {:?}", a, b, result, stack);
                }
                
                WasmInstruction::F64Sub => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    let result = a - b;
                    stack.push(result);
                    println!("{} - {} = {} -> Stack: {:?}", a, b, result, stack);
                }
                
                WasmInstruction::F64Mul => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap(); 
                    let result = a * b;
                    stack.push(result);
                    println!("{} * {} = {} -> Stack: {:?}", a, b, result, stack);
                }
                
                WasmInstruction::F64Div => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    let result = a / b;
                    stack.push(result);
                    println!("{} / {} = {} -> Stack: {:?}", a, b, result, stack);
                }
                
                WasmInstruction::Call(func_name) => {
                    match func_name.as_str() {
                        "sin" => {
                            let arg = stack.pop().unwrap();
                            let result = arg.sin();
                            stack.push(result);
                            println!("sin({}) = {} -> Stack: {:?}", arg, result, stack);
                        }
                        "cos" => {
                            let arg = stack.pop().unwrap();
                            let result = arg.cos();
                            stack.push(result);
                            println!("cos({}) = {} -> Stack: {:?}", arg, result, stack);
                        }
                        "sqrt" => {
                            let arg = stack.pop().unwrap();
                            let result = arg.sqrt();
                            stack.push(result);
                            println!("sqrt({}) = {} -> Stack: {:?}", arg, result, stack);
                        }
                        "pow" => {
                            let exponent = stack.pop().unwrap();
                            let base = stack.pop().unwrap();
                            let result = base.powf(exponent);
                            stack.push(result);
                            println!("pow({}, {}) = {} -> Stack: {:?}", base, exponent, result, stack);
                        }
                        _ => println!("Unknown function: {}", func_name),
                    }
                }
            }
        }
        
        let final_result = stack.pop().unwrap();
        println!("\n✅ Final result: {}", final_result);
        
        final_result
    }
}

/// Complete Example: Full compilation pipeline
pub fn demonstrate_complete_pipeline() {
    println!("🎯 COMPLETE MATHEMATICAL COMPILATION PIPELINE");
    println!("==============================================\n");
    
    let input_expression = "x^2 + sin(y * 3.14159)";
    let variables = vec!["x".to_string(), "y".to_string()];
    let input_values = vec![3.0, 0.5]; // x=3, y=0.5
    
    println!("📝 Input Expression: {}", input_expression);
    println!("📝 Variables: {:?}", variables);
    println!("📝 Input Values: {:?}", input_values);
    
    // Step 1: Tokenization
    let mut tokenizer = DetailedTokenizer {
        input: input_expression.to_string(),
        position: 0,
    };
    let tokens = tokenizer.tokenize_step_by_step();
    
    // Step 2: Parsing
    let mut parser = DetailedParser {
        tokens,
        position: 0,
    };
    let ast = parser.parse_with_explanation();
    
    // Step 3: Optimization
    let mut optimizer = DetailedOptimizer {
        optimization_level: OptimizationLevel::Aggressive,
    };
    let optimized_ast = optimizer.optimize_with_explanation(ast);
    
    // Step 4: Code Generation
    let mut code_generator = DetailedCodeGenerator {
        wasm_instructions: Vec::new(),
        local_variables: Vec::new(),
    };
    let instructions = code_generator.generate_with_explanation(&optimized_ast, &variables);
    
    // Step 5: Runtime Execution
    let mut runtime = DetailedRuntime {
        compiled_functions: std::collections::HashMap::new(),
    };
    runtime.compiled_functions.insert("test_function".to_string(), instructions);
    
    let result = runtime.execute_with_explanation("test_function", &input_values);
    
    println!("\n🎉 COMPILATION COMPLETE!");
    println!("Expression: {} with x={}, y={}", input_expression, input_values[0], input_values[1]);
    println!("Result: {}", result);
    
    // Verify with direct calculation
    let expected = input_values[0].powf(2.0) + (input_values[1] * 3.14159).sin();
    println!("Expected: {}", expected);
    println!("Difference: {}", (result - expected).abs());
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complete_pipeline() {
        demonstrate_complete_pipeline();
    }
}
