use wasm_bindgen_test::*;
use math_compiler_wasm::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_basic_arithmetic() {
    let mut platform = MathCompilerPlatform::new();
    
    // Test simple addition
    let result = platform.evaluate_expression(
        "x + y",
        vec!["x".to_string(), "y".to_string()],
        vec![5.0, 3.0]
    ).unwrap();
    
    assert!((result - 8.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_mathematical_functions() {
    let mut platform = MathCompilerPlatform::new();
    
    // Test trigonometric functions
    let result = platform.evaluate_expression(
        "sin(x) + cos(y)",
        vec!["x".to_string(), "y".to_string()],
        vec![0.0, 0.0]
    ).unwrap();
    
    // sin(0) + cos(0) = 0 + 1 = 1
    assert!((result - 1.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_complex_expressions() {
    let mut platform = MathCompilerPlatform::new();
    
    // Test complex mathematical expression
    let result = platform.evaluate_expression(
        "sqrt(x^2 + y^2)",
        vec!["x".to_string(), "y".to_string()],
        vec![3.0, 4.0]
    ).unwrap();
    
    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    assert!((result - 5.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_power_operations() {
    let mut platform = MathCompilerPlatform::new();
    
    let result = platform.evaluate_expression(
        "pow(x, y)",
        vec!["x".to_string(), "y".to_string()],
        vec![2.0, 3.0]
    ).unwrap();
    
    // 2^3 = 8
    assert!((result - 8.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_multiple_models() {
    let mut platform = MathCompilerPlatform::new();
    
    // Compile first model
    let model1 = platform.compile_model(
        "model1",
        "Linear Function",
        "2 * x + 1",
        vec!["x".to_string()]
    ).unwrap();
    
    // Compile second model
    let model2 = platform.compile_model(
        "model2", 
        "Quadratic Function",
        "x^2 + 2*x + 1",
        vec!["x".to_string()]
    ).unwrap();
    
    // Load both models
    platform.load_model("model1").unwrap();
    platform.load_model("model2").unwrap();
    
    // Test first model: 2*5 + 1 = 11
    let result1 = platform.execute_model("model1", vec![5.0]).unwrap();
    assert!((result1 - 11.0).abs() < 0.0001);
    
    // Test second model: 5^2 + 2*5 + 1 = 25 + 10 + 1 = 36
    let result2 = platform.execute_model("model2", vec![5.0]).unwrap();
    assert!((result2 - 36.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_expression_validation() {
    let platform = MathCompilerPlatform::new();
    
    // Valid expression
    let valid_result = platform.validate_expression("x + y * 2").unwrap();
    assert!(js_sys::Reflect::get(&valid_result, &"valid".into()).unwrap().as_bool().unwrap());
    
    // Invalid expression (missing operand)
    let invalid_result = platform.validate_expression("x + * 2").unwrap();
    assert!(!js_sys::Reflect::get(&invalid_result, &"valid".into()).unwrap().as_bool().unwrap());
}

#[wasm_bindgen_test]
fn test_model_management() {
    let mut platform = MathCompilerPlatform::new();
    
    // Compile a model
    platform.compile_model(
        "test_model",
        "Test Model",
        "x * 2",
        vec!["x".to_string()]
    ).unwrap();
    
    // Check model exists in list
    let models = platform.list_models().unwrap();
    let models_array: js_sys::Array = models.into();
    assert_eq!(models_array.length(), 1);
    
    // Remove model
    platform.remove_model("test_model").unwrap();
    
    // Check model list is empty
    let models_after = platform.list_models().unwrap();
    let models_array_after: js_sys::Array = models_after.into();
    assert_eq!(models_array_after.length(), 0);
}

#[wasm_bindgen_test]
fn test_error_handling() {
    let mut platform = MathCompilerPlatform::new();
    
    // Test division by zero handling
    let result = platform.evaluate_expression(
        "x / y",
        vec!["x".to_string(), "y".to_string()],
        vec![1.0, 0.0]
    );
    
    // Should handle gracefully (return infinity in WebAssembly)
    assert!(result.is_ok());
    assert!(result.unwrap().is_infinite());
}

#[wasm_bindgen_test]
fn test_performance_benchmark() {
    let mut platform = MathCompilerPlatform::new();
    
    // Compile a moderately complex expression
    platform.compile_model(
        "perf_test",
        "Performance Test",
        "sin(x) * cos(y) + sqrt(x^2 + y^2)",
        vec!["x".to_string(), "y".to_string()]
    ).unwrap();
    
    platform.load_model("perf_test").unwrap();
    
    // Benchmark execution time
    let iterations = 1000;
    let start_time = js_sys::Date::now();
    
    for _ in 0..iterations {
        platform.execute_model("perf_test", vec![1.0, 2.0]).unwrap();
    }
    
    let end_time = js_sys::Date::now();
    let total_time = end_time - start_time;
    let avg_time = total_time / iterations as f64;
    
    web_sys::console::log_2(
        &"Performance benchmark:".into(),
        &format!("Average execution time: {:.4}ms", avg_time).into()
    );
    
    // Verify performance is reasonable (less than 1ms per execution)
    assert!(avg_time < 1.0);
}

#[wasm_bindgen_test]
fn test_parser_comprehensive() {
    let parser = MathParser::new();
    
    let test_cases = vec![
        ("x + y", true),
        ("sin(x) * cos(y)", true), 
        ("sqrt(x^2 + y^2)", true),
        ("if(x > 0) x else -x", true),
        ("max(min(x, y), 0)", true),
        ("x +", false), // Invalid: missing operand
        ("sin()", false), // Invalid: missing argument
        ("x ^ ^ y", false), // Invalid: double operator
        ("", false), // Invalid: empty expression
    ];
    
    for (expression, should_be_valid) in test_cases {
        let result = parser.parse_expression(expression);
        
        if should_be_valid {
            assert!(result.is_ok(), "Expression '{}' should be valid", expression);
        } else {
            assert!(result.is_err(), "Expression '{}' should be invalid", expression);
        }
    }
}

#[wasm_bindgen_test]
fn test_variable_scoping() {
    let mut platform = MathCompilerPlatform::new();
    
    // Test with different variable names and counts
    let single_var = platform.evaluate_expression(
        "x^3",
        vec!["x".to_string()],
        vec![2.0]
    ).unwrap();
    assert!((single_var - 8.0).abs() < 0.0001);
    
    let multi_var = platform.evaluate_expression(
        "a * b + c",
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![2.0, 3.0, 1.0]
    ).unwrap();
    assert!((multi_var - 7.0).abs() < 0.0001);
}

#[wasm_bindgen_test]
fn test_numerical_precision() {
    let mut platform = MathCompilerPlatform::new();
    
    // Test with very small numbers
    let small_result = platform.evaluate_expression(
        "x + y",
        vec!["x".to_string(), "y".to_string()],
        vec![1e-10, 2e-10]
    ).unwrap();
    assert!((small_result - 3e-10).abs() < 1e-15);
    
    // Test with very large numbers
    let large_result = platform.evaluate_expression(
        "x * y", 
        vec!["x".to_string(), "y".to_string()],
        vec![1e10, 2e10]
    ).unwrap();
    assert!((large_result - 2e20).abs() < 1e15);
}
