use pest::Parser;
use pest_derive::Parser;
use pest::iterators::{Pair, Pairs};
use crate::ast::*;
use wasm_bindgen::prelude::*;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "math_expression.pest"]
pub struct MathExpressionParser;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Pest parsing error: {0}")]
    PestError(#[from] pest::error::Error<Rule>),
    #[error("Invalid function name: {0}")]
    InvalidFunction(String),
    #[error("Invalid number of arguments for function {function}: expected {expected}, got {actual}")]
    InvalidArity { function: String, expected: usize, actual: usize },
    #[error("Unknown identifier: {0}")]
    UnknownIdentifier(String),
}

#[wasm_bindgen]
pub struct MathParser {
    // We could add configuration options here later
}

#[wasm_bindgen]
impl MathParser {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MathParser {
        MathParser {}
    }
    
    #[wasm_bindgen]
    pub fn parse_expression(&self, input: &str) -> Result<JsValue, JsValue> {
        match self.parse_expression_internal(input) {
            Ok(expr) => {
                serde_wasm_bindgen::to_value(&expr).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            Err(e) => Err(JsValue::from_str(&e.to_string()))
        }
    }
    
    #[wasm_bindgen]
    pub fn parse_model(&self, input: &str) -> Result<JsValue, JsValue> {
        match self.parse_model_internal(input) {
            Ok(model) => {
                serde_wasm_bindgen::to_value(&model).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            Err(e) => Err(JsValue::from_str(&e.to_string()))
        }
    }
}

impl MathParser {
    pub fn parse_expression_internal(&self, input: &str) -> Result<MathExpr, ParseError> {
        let mut pairs = MathExpressionParser::parse(Rule::math_input, input)?;
        let pair = pairs.next().unwrap();
        
        match pair.as_rule() {
            Rule::math_input => {
                let inner = pair.into_inner().next().unwrap();
                match inner.as_rule() {
                    Rule::expression => self.parse_expression_pair(inner),
                    Rule::model => {
                        let model = self.parse_model_pair(inner)?;
                        if model.statements.len() == 1 {
                            match &model.statements[0] {
                                Statement::Expression(expr) => Ok(expr.clone()),
                                _ => Err(ParseError::UnknownIdentifier("Expected single expression".to_string())),
                            }
                        } else {
                            Err(ParseError::UnknownIdentifier("Expected single expression".to_string()))
                        }
                    }
                    _ => unreachable!()
                }
            }
            _ => unreachable!()
        }
    }
    
    pub fn parse_model_internal(&self, input: &str) -> Result<MathModel, ParseError> {
        let mut pairs = MathExpressionParser::parse(Rule::math_input, input)?;
        let pair = pairs.next().unwrap();
        
        match pair.as_rule() {
            Rule::math_input => {
                let inner = pair.into_inner().next().unwrap();
                match inner.as_rule() {
                    Rule::model => self.parse_model_pair(inner),
                    Rule::expression => {
                        let expr = self.parse_expression_pair(inner)?;
                        let mut model = MathModel::new();
                        model.add_statement(Statement::Expression(expr));
                        Ok(model)
                    }
                    _ => unreachable!()
                }
            }
            _ => unreachable!()
        }
    }
    
    fn parse_model_pair(&self, pair: Pair<Rule>) -> Result<MathModel, ParseError> {
        let mut model = MathModel::new();
        
        for statement_pair in pair.into_inner() {
            match statement_pair.as_rule() {
                Rule::statement => {
                    let inner = statement_pair.into_inner().next().unwrap();
                    let statement = match inner.as_rule() {
                        Rule::assignment => {
                            let mut assignment_pairs = inner.into_inner();
                            let variable = assignment_pairs.next().unwrap().as_str().to_string();
                            let expression = self.parse_expression_pair(assignment_pairs.next().unwrap())?;
                            Statement::Assignment(Assignment { variable, expression })
                        }
                        Rule::expression => {
                            let expression = self.parse_expression_pair(inner)?;
                            Statement::Expression(expression)
                        }
                        _ => unreachable!()
                    };
                    model.add_statement(statement);
                }
                Rule::EOI => break,
                _ => {}
            }
        }
        
        Ok(model)
    }
    
    fn parse_expression_pair(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        match pair.as_rule() {
            Rule::expression => {
                let inner = pair.into_inner().next().unwrap();
                self.parse_logical_or(inner)
            }
            _ => self.parse_logical_or(pair)
        }
    }
    
    fn parse_logical_or(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let mut left = self.parse_logical_and(pairs.next().unwrap())?;
        
        while let Some(op_pair) = pairs.next() {
            if op_pair.as_rule() == Rule::or {
                let right = self.parse_logical_and(pairs.next().unwrap())?;
                left = MathExpr::BinaryOp {
                    op: BinaryOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            }
        }
        
        Ok(left)
    }
    
    fn parse_logical_and(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let mut left = self.parse_comparison(pairs.next().unwrap())?;
        
        while let Some(op_pair) = pairs.next() {
            if op_pair.as_rule() == Rule::and {
                let right = self.parse_comparison(pairs.next().unwrap())?;
                left = MathExpr::BinaryOp {
                    op: BinaryOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            }
        }
        
        Ok(left)
    }
    
    fn parse_comparison(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let mut left = self.parse_additive(pairs.next().unwrap())?;
        
        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_rule() {
                Rule::eq => BinaryOp::Equal,
                Rule::ne => BinaryOp::NotEqual,
                Rule::lt => BinaryOp::LessThan,
                Rule::le => BinaryOp::LessEqual,
                Rule::gt => BinaryOp::GreaterThan,
                Rule::ge => BinaryOp::GreaterEqual,
                _ => continue,
            };
            
            let right = self.parse_additive(pairs.next().unwrap())?;
            left = MathExpr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    fn parse_additive(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let mut left = self.parse_multiplicative(pairs.next().unwrap())?;
        
        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_rule() {
                Rule::add => BinaryOp::Add,
                Rule::subtract => BinaryOp::Subtract,
                _ => continue,
            };
            
            let right = self.parse_multiplicative(pairs.next().unwrap())?;
            left = MathExpr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    fn parse_multiplicative(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let mut left = self.parse_power_expr(pairs.next().unwrap())?;
        
        while let Some(op_pair) = pairs.next() {
            let op = match op_pair.as_rule() {
                Rule::multiply => BinaryOp::Multiply,
                Rule::divide => BinaryOp::Divide,
                Rule::modulo => BinaryOp::Modulo,
                _ => continue,
            };
            
            let right = self.parse_power_expr(pairs.next().unwrap())?;
            left = MathExpr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    fn parse_power_expr(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let mut left = self.parse_unary(pairs.next().unwrap())?;
        
        while let Some(op_pair) = pairs.next() {
            if op_pair.as_rule() == Rule::power {
                let right = self.parse_unary(pairs.next().unwrap())?;
                left = MathExpr::BinaryOp {
                    op: BinaryOp::Power,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            }
        }
        
        Ok(left)
    }
    
    fn parse_unary(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        let mut pairs = pair.into_inner();
        let first = pairs.next().unwrap();
        
        match first.as_rule() {
            Rule::subtract => {
                let operand = self.parse_primary(pairs.next().unwrap())?;
                Ok(MathExpr::UnaryOp {
                    op: UnaryOp::Negate,
                    operand: Box::new(operand),
                })
            }
            Rule::not => {
                let operand = self.parse_primary(pairs.next().unwrap())?;
                Ok(MathExpr::UnaryOp {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_primary(first)
        }
    }
    
    fn parse_primary(&self, pair: Pair<Rule>) -> Result<MathExpr, ParseError> {
        match pair.as_rule() {
            Rule::number => {
                let value: f64 = pair.as_str().parse().unwrap();
                Ok(MathExpr::Number(value))
            }
            Rule::identifier => {
                Ok(MathExpr::Identifier(pair.as_str().to_string()))
            }
            Rule::function_call => {
                let mut pairs = pair.into_inner();
                let function_name = pairs.next().unwrap().as_str();
                let function = MathFunction::from_str(function_name)
                    .ok_or_else(|| ParseError::InvalidFunction(function_name.to_string()))?;
                
                let mut args = Vec::new();
                for arg_pair in pairs {
                    args.push(self.parse_expression_pair(arg_pair)?);
                }
                
                // Validate arity for most functions (some like min/max can be variable)
                let expected_arity = function.arity();
                if expected_arity > 0 && args.len() != expected_arity {
                    return Err(ParseError::InvalidArity {
                        function: function_name.to_string(),
                        expected: expected_arity,
                        actual: args.len(),
                    });
                }
                
                Ok(MathExpr::FunctionCall { function, args })
            }
            Rule::array_access => {
                let mut pairs = pair.into_inner();
                let array_name = pairs.next().unwrap().as_str().to_string();
                let index = self.parse_expression_pair(pairs.next().unwrap())?;
                
                Ok(MathExpr::ArrayAccess {
                    array: array_name,
                    index: Box::new(index),
                })
            }
            Rule::conditional => {
                let mut pairs = pair.into_inner();
                let condition = self.parse_expression_pair(pairs.next().unwrap())?;
                let true_expr = self.parse_expression_pair(pairs.next().unwrap())?;
                let false_expr = self.parse_expression_pair(pairs.next().unwrap())?;
                
                Ok(MathExpr::Conditional {
                    condition: Box::new(condition),
                    true_expr: Box::new(true_expr),
                    false_expr: Box::new(false_expr),
                })
            }
            Rule::expression => self.parse_expression_pair(pair),
            _ => unreachable!("Unexpected rule: {:?}", pair.as_rule())
        }
    }
}
