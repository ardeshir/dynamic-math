use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,
    // Comparison operators
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    // Logical operators
    And,
    Or,
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathFunction {
    // Trigonometric functions
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    
    // Logarithmic and exponential
    Log,
    Ln,
    Exp,
    
    // Basic math
    Sqrt,
    Abs,
    Ceil,
    Floor,
    Round,
    
    // Multi-argument functions
    Min,
    Max,
    Pow,
    
    // Formulation-specific functions
    NutritionalValue,
    CostPerUnit,
    Digestibility,
    Bioavailability,
    InteractionFactor,
}

impl MathFunction {
    pub fn from_str(s: &str) -> Option<MathFunction> {
        match s {
            "sin" => Some(MathFunction::Sin),
            "cos" => Some(MathFunction::Cos),
            "tan" => Some(MathFunction::Tan),
            "asin" => Some(MathFunction::Asin),
            "acos" => Some(MathFunction::Acos),
            "atan" => Some(MathFunction::Atan),
            "log" => Some(MathFunction::Log),
            "ln" => Some(MathFunction::Ln),
            "exp" => Some(MathFunction::Exp),
            "sqrt" => Some(MathFunction::Sqrt),
            "abs" => Some(MathFunction::Abs),
            "ceil" => Some(MathFunction::Ceil),
            "floor" => Some(MathFunction::Floor),
            "round" => Some(MathFunction::Round),
            "min" => Some(MathFunction::Min),
            "max" => Some(MathFunction::Max),
            "pow" => Some(MathFunction::Pow),
            "nutritional_value" => Some(MathFunction::NutritionalValue),
            "cost_per_unit" => Some(MathFunction::CostPerUnit),
            "digestibility" => Some(MathFunction::Digestibility),
            "bioavailability" => Some(MathFunction::Bioavailability),
            "interaction_factor" => Some(MathFunction::InteractionFactor),
            _ => None,
        }
    }
    
    pub fn arity(&self) -> usize {
        match self {
            MathFunction::Sin | MathFunction::Cos | MathFunction::Tan |
            MathFunction::Asin | MathFunction::Acos | MathFunction::Atan |
            MathFunction::Log | MathFunction::Ln | MathFunction::Exp |
            MathFunction::Sqrt | MathFunction::Abs | MathFunction::Ceil |
            MathFunction::Floor | MathFunction::Round => 1,
            
            MathFunction::Min | MathFunction::Max | MathFunction::Pow => 2,
            
            // Formulation functions - context dependent
            MathFunction::NutritionalValue => 2, // (ingredient, nutrient)
            MathFunction::CostPerUnit => 1,      // (ingredient)
            MathFunction::Digestibility => 2,    // (ingredient, species)
            MathFunction::Bioavailability => 2,  // (nutrient, ingredient)
            MathFunction::InteractionFactor => 2, // (ingredient1, ingredient2)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MathExpr {
    // Literals
    Number(f64),
    Identifier(String),
    
    // Operations
    BinaryOp {
        op: BinaryOp,
        left: Box<MathExpr>,
        right: Box<MathExpr>,
    },
    
    UnaryOp {
        op: UnaryOp,
        operand: Box<MathExpr>,
    },
    
    // Function calls
    FunctionCall {
        function: MathFunction,
        args: Vec<MathExpr>,
    },
    
    // Array/vector access
    ArrayAccess {
        array: String,
        index: Box<MathExpr>,
    },
    
    // Conditional expressions
    Conditional {
        condition: Box<MathExpr>,
        true_expr: Box<MathExpr>,
        false_expr: Box<MathExpr>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    pub variable: String,
    pub expression: MathExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Assignment(Assignment),
    Expression(MathExpr),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MathModel {
    pub statements: Vec<Statement>,
}

impl MathModel {
    pub fn new() -> Self {
        MathModel {
            statements: Vec::new(),
        }
    }
    
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }
    
    pub fn get_variables(&self) -> Vec<String> {
        let mut variables = std::collections::HashSet::new();
        
        for statement in &self.statements {
            match statement {
                Statement::Assignment(assignment) => {
                    variables.insert(assignment.variable.clone());
                    self.collect_variables(&assignment.expression, &mut variables);
                }
                Statement::Expression(expr) => {
                    self.collect_variables(expr, &mut variables);
                }
            }
        }
        
        variables.into_iter().collect()
    }
    
    fn collect_variables(&self, expr: &MathExpr, variables: &mut std::collections::HashSet<String>) {
        match expr {
            MathExpr::Identifier(name) => {
                variables.insert(name.clone());
            }
            MathExpr::BinaryOp { left, right, .. } => {
                self.collect_variables(left, variables);
                self.collect_variables(right, variables);
            }
            MathExpr::UnaryOp { operand, .. } => {
                self.collect_variables(operand, variables);
            }
            MathExpr::FunctionCall { args, .. } => {
                for arg in args {
                    self.collect_variables(arg, variables);
                }
            }
            MathExpr::ArrayAccess { array, index } => {
                variables.insert(array.clone());
                self.collect_variables(index, variables);
            }
            MathExpr::Conditional { condition, true_expr, false_expr } => {
                self.collect_variables(condition, variables);
                self.collect_variables(true_expr, variables);
                self.collect_variables(false_expr, variables);
            }
            MathExpr::Number(_) => {}
        }
    }
}

// Variable context for evaluation and compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableContext {
    pub variables: HashMap<String, f64>,
    pub arrays: HashMap<String, Vec<f64>>,
}

impl VariableContext {
    pub fn new() -> Self {
        VariableContext {
            variables: HashMap::new(),
            arrays: HashMap::new(),
        }
    }
    
    pub fn set_variable(&mut self, name: String, value: f64) {
        self.variables.insert(name, value);
    }
    
    pub fn set_array(&mut self, name: String, values: Vec<f64>) {
        self.arrays.insert(name, values);
    }
    
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
    }
    
    pub fn get_array(&self, name: &str) -> Option<&Vec<f64>> {
        self.arrays.get(name)
    }
}
