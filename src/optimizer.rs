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
