/**
 * Comprehensive JavaScript Integration Example
 * Dynamic Mathematical Compilation Platform
 * 
 * This example demonstrates how to integrate the math compiler
 * into various JavaScript environments and use cases.
 */

import init, { 
    MathCompilerPlatform, 
    WasmExpressionCache,
    set_panic_hook 
} from './pkg/math_compiler_wasm.js';

/**
 * High-level Mathematical Expression Compiler
 * Provides a clean API with caching, optimization, and error handling
 */
class AdvancedMathCompiler {
    constructor(options = {}) {
        this.platform = null;
        this.cache = null;
        this.isInitialized = false;
        this.options = {
            enableCache: true,
            maxCacheEntries: 100,
            maxCacheSizeMB: 10,
            optimizationLevel: 'basic',
            enableLogging: false,
            ...options
        };
        
        this.compilationStats = {
            totalCompilations: 0,
            cacheHits: 0,
            cacheMisses: 0,
            totalExecutions: 0,
            averageCompilationTime: 0,
            averageExecutionTime: 0
        };
    }
    
    /**
     * Initialize the compiler platform
     */
    async initialize() {
        if (this.isInitialized) return;
        
        try {
            await init();
            set_panic_hook();
            
            this.platform = new MathCompilerPlatform();
            
            if (this.options.enableCache) {
                this.cache = new WasmExpressionCache(
                    this.options.maxCacheEntries,
                    this.options.maxCacheSizeMB
                );
            }
            
            this.isInitialized = true;
            this.log('Math Compiler Platform initialized successfully');
        } catch (error) {
            throw new Error(`Failed to initialize math compiler: ${error.message}`);
        }
    }
    
    /**
     * Compile and execute a mathematical expression
     * @param {string} expression - Mathematical expression
     * @param {Object} variables - Variable values as key-value pairs
     * @param {Object} options - Compilation options
     * @returns {Promise<number>} Result of the expression
     */
    async evaluate(expression, variables = {}, options = {}) {
        await this.ensureInitialized();
        
        const startTime = performance.now();
        
        try {
            const variableNames = Object.keys(variables);
            const variableValues = Object.values(variables);
            
            // Generate cache key
            const cacheKey = this.generateCacheKey(expression, variableNames, options);
            let modelId = null;
            
            // Check cache first
            if (this.cache && this.cache.has_entry(cacheKey)) {
                modelId = cacheKey;
                this.compilationStats.cacheHits++;
                this.log(`Cache hit for expression: ${expression}`);
            } else {
                // Compile new expression
                const compilationStart = performance.now();
                
                modelId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                
                await this.platform.compile_model(
                    modelId,
                    `Generated Model`,
                    expression,
                    variableNames
                );
                
                await this.platform.load_model(modelId);
                
                const compilationTime = performance.now() - compilationStart;
                this.updateCompilationStats(compilationTime);
                
                this.compilationStats.cacheMisses++;
                this.log(`Compiled expression: ${expression} (${compilationTime.toFixed(2)}ms)`);
            }
            
            // Execute the model
            const result = await this.platform.execute_model(modelId, variableValues);
            
            const totalTime = performance.now() - startTime;
            this.updateExecutionStats(totalTime);
            
            return result;
            
        } catch (error) {
            throw new Error(`Expression evaluation failed: ${error.message}`);
        }
    }
    
    /**
     * Batch evaluate multiple expressions with the same variables
     * @param {Array<string>} expressions - Array of mathematical expressions
     * @param {Object} variables - Variable values
     * @returns {Promise<Array<number>>} Array of results
     */
    async evaluateBatch(expressions, variables = {}) {
        await this.ensureInitialized();
        
        const results = [];
        
        for (const expression of expressions) {
            try {
                const result = await this.evaluate(expression, variables);
                results.push(result);
            } catch (error) {
                this.log(`Error evaluating '${expression}': ${error.message}`);
                results.push(NaN);
            }
        }
        
        return results;
    }
    
    /**
     * Validate a mathematical expression
     * @param {string} expression - Expression to validate
     * @returns {Promise<Object>} Validation result
     */
    async validate(expression) {
        await this.ensureInitialized();
        
        try {
            const result = await this.platform.validate_expression(expression);
            return {
                valid: result.valid,
                error: result.error,
                ast: result.ast
            };
        } catch (error) {
            return {
                valid: false,
                error: error.message,
                ast: null
            };
        }
    }
    
    /**
     * Create a reusable compiled function
     * @param {string} expression - Mathematical expression
     * @param {Array<string>} variableNames - Variable names in order
     * @returns {Promise<Function>} Compiled function
     */
    async compileFunction(expression, variableNames = []) {
        await this.ensureInitialized();
        
        const modelId = `func_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        await this.platform.compile_model(
            modelId,
            `Compiled Function`,
            expression,
            variableNames
        );
        
        await this.platform.load_model(modelId);
        
        // Return a closure that maintains the model ID
        return async (...args) => {
            if (args.length !== variableNames.length) {
                throw new Error(
                    `Expected ${variableNames.length} arguments, got ${args.length}`
                );
            }
            
            try {
                return await this.platform.execute_model(modelId, args);
            } catch (error) {
                throw new Error(`Function execution failed: ${error.message}`);
            }
        };
    }
    
    /**
     * Performance benchmark for an expression
     * @param {string} expression - Expression to benchmark
     * @param {Object} variables - Variable values
     * @param {number} iterations - Number of iterations
     * @returns {Promise<Object>} Benchmark results
     */
    async benchmark(expression, variables = {}, iterations = 1000) {
        await this.ensureInitialized();
        
        const variableNames = Object.keys(variables);
        const variableValues = Object.values(variables);
        
        // Compile once
        const modelId = `bench_${Date.now()}`;
        const compilationStart = performance.now();
        
        await this.platform.compile_model(
            modelId,
            'Benchmark Model',
            expression,
            variableNames
        );
        await this.platform.load_model(modelId);
        
        const compilationTime = performance.now() - compilationStart;
        
        // Warm-up runs
        for (let i = 0; i < 10; i++) {
            await this.platform.execute_model(modelId, variableValues);
        }
        
        // Benchmark runs
        const times = [];
        let totalTime = 0;
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await this.platform.execute_model(modelId, variableValues);
            const time = performance.now() - start;
            times.push(time);
            totalTime += time;
        }
        
        // Calculate statistics
        times.sort((a, b) => a - b);
        const averageTime = totalTime / iterations;
        const medianTime = times[Math.floor(iterations / 2)];
        const minTime = times[0];
        const maxTime = times[iterations - 1];
        const execPerSecond = 1000 / averageTime;
        
        // Clean up
        await this.platform.remove_model(modelId);
        
        return {
            expression,
            iterations,
            compilationTimeMs: compilationTime,
            totalExecutionTimeMs: totalTime,
            averageTimeMs: averageTime,
            medianTimeMs: medianTime,
            minTimeMs: minTime,
            maxTimeMs: maxTime,
            executionsPerSecond: Math.round(execPerSecond),
            variables: variableNames
        };
    }
    
    /**
     * Get comprehensive statistics
     * @returns {Promise<Object>} Statistics
     */
    async getStatistics() {
        const stats = { ...this.compilationStats };
        
        if (this.cache) {
            const cacheStats = await this.cache.get_stats();
            stats.cache = cacheStats;
            stats.cacheRecommendations = this.cache.get_recommendations();
            stats.cacheStorageUsageMB = this.cache.estimate_storage_usage();
        }
        
        if (this.platform) {
            const models = await this.platform.list_models();
            stats.activeModels = models.length;
        }
        
        return stats;
    }
    
    /**
     * Clear all compiled models and cache
     */
    async clear() {
        if (this.platform) {
            const models = await this.platform.list_models();
            for (const model of models) {
                await this.platform.remove_model(model.id);
            }
        }
        
        if (this.cache) {
            this.cache.clear_cache();
        }
        
        // Reset statistics
        this.compilationStats = {
            totalCompilations: 0,
            cacheHits: 0,
            cacheMisses: 0,
            totalExecutions: 0,
            averageCompilationTime: 0,
            averageExecutionTime: 0
        };
    }
    
    /**
     * Perform maintenance operations
     */
    async maintenance() {
        if (this.cache) {
            this.cache.maintenance();
        }
        
        this.log('Maintenance completed');
    }
    
    // Private helper methods
    
    async ensureInitialized() {
        if (!this.isInitialized) {
            await this.initialize();
        }
    }
    
    generateCacheKey(expression, variableNames, options) {
        if (!this.cache) return null;
        
        return WasmExpressionCache.generate_key(
            expression,
            variableNames,
            options.optimizationLevel || this.options.optimizationLevel
        );
    }
    
    updateCompilationStats(compilationTime) {
        this.compilationStats.totalCompilations++;
        
        const total = this.compilationStats.averageCompilationTime * 
                     (this.compilationStats.totalCompilations - 1) + compilationTime;
        this.compilationStats.averageCompilationTime = 
            total / this.compilationStats.totalCompilations;
    }
    
    updateExecutionStats(executionTime) {
        this.compilationStats.totalExecutions++;
        
        const total = this.compilationStats.averageExecutionTime * 
                     (this.compilationStats.totalExecutions - 1) + executionTime;
        this.compilationStats.averageExecutionTime = 
            total / this.compilationStats.totalExecutions;
    }
    
    log(message) {
        if (this.options.enableLogging) {
            console.log(`[MathCompiler] ${message}`);
        }
    }
}

/**
 * Specialized classes for different use cases
 */

/**
 * Real-time Mathematical Expression Evaluator
 * Optimized for frequent evaluations with changing variables
 */
class RealTimeMathEvaluator extends AdvancedMathCompiler {
    constructor(options = {}) {
        super({
            enableCache: true,
            maxCacheEntries: 50,
            maxCacheSizeMB: 5,
            optimizationLevel: 'aggressive',
            ...options
        });
        
        this.activeExpressions = new Map();
        this.evaluationQueue = [];
        this.isProcessingQueue = false;
    }
    
    /**
     * Register an expression for real-time evaluation
     * @param {string} id - Unique identifier for the expression
     * @param {string} expression - Mathematical expression
     * @param {Array<string>} variableNames - Variable names
     */
    async registerExpression(id, expression, variableNames = []) {
        await this.ensureInitialized();
        
        const modelId = `rt_${id}`;
        
        await this.platform.compile_model(
            modelId,
            `RealTime: ${id}`,
            expression,
            variableNames
        );
        
        await this.platform.load_model(modelId);
        
        this.activeExpressions.set(id, {
            modelId,
            expression,
            variableNames,
            lastEvaluation: null,
            evaluationCount: 0
        });
        
        this.log(`Registered real-time expression: ${id} -> ${expression}`);
    }
    
    /**
     * Evaluate a registered expression
     * @param {string} id - Expression ID
     * @param {Array<number>} values - Variable values
     * @returns {Promise<number>} Result
     */
    async evaluateById(id, values) {
        const expr = this.activeExpressions.get(id);
        if (!expr) {
            throw new Error(`Expression '${id}' not registered`);
        }
        
        try {
            const result = await this.platform.execute_model(expr.modelId, values);
            
            expr.lastEvaluation = {
                timestamp: Date.now(),
                values,
                result
            };
            expr.evaluationCount++;
            
            return result;
        } catch (error) {
            throw new Error(`Real-time evaluation failed for '${id}': ${error.message}`);
        }
    }
    
    /**
     * Batch evaluate multiple registered expressions
     * @param {Array<{id: string, values: Array<number>}>} evaluations
     * @returns {Promise<Array<{id: string, result: number, error?: string}>>}
     */
    async evaluateBatchById(evaluations) {
        const results = [];
        
        for (const { id, values } of evaluations) {
            try {
                const result = await this.evaluateById(id, values);
                results.push({ id, result });
            } catch (error) {
                results.push({ id, result: NaN, error: error.message });
            }
        }
        
        return results;
    }
    
    /**
     * Get real-time statistics for registered expressions
     */
    getRealTimeStats() {
        const stats = {};
        
        for (const [id, expr] of this.activeExpressions) {
            stats[id] = {
                expression: expr.expression,
                variableNames: expr.variableNames,
                evaluationCount: expr.evaluationCount,
                lastEvaluation: expr.lastEvaluation
            };
        }
        
        return stats;
    }
}

/**
 * Formulation-specific Mathematical Compiler
 * Specialized for feed/chemical formulation calculations
 */
class FormulationMathCompiler extends AdvancedMathCompiler {
    constructor(options = {}) {
        super({
            enableCache: true,
            optimizationLevel: 'aggressive',
            ...options
        });
        
        this.ingredientDatabase = new Map();
        this.nutrientDatabase = new Map();
        this.formulationModels = new Map();
    }
    
    /**
     * Register an ingredient with its nutritional profile
     * @param {string} name - Ingredient name
     * @param {Object} nutrition - Nutritional data
     */
    registerIngredient(name, nutrition) {
        this.ingredientDatabase.set(name, nutrition);
        this.log(`Registered ingredient: ${name}`);
    }
    
    /**
     * Register a nutrient with its properties
     * @param {string} name - Nutrient name
     * @param {Object} properties - Nutrient properties
     */
    registerNutrient(name, properties) {
        this.nutrientDatabase.set(name, properties);
        this.log(`Registered nutrient: ${name}`);
    }
    
    /**
     * Create a formulation model with constraints
     * @param {string} id - Formulation ID
     * @param {Object} config - Formulation configuration
     */
    async createFormulation(id, config) {
        await this.ensureInitialized();
        
        const {
            objective,           // e.g., "minimize cost"
            constraints,         // Array of constraint expressions
            ingredients,         // Available ingredients
            targets             // Nutritional targets
        } = config;
        
        // Build the formulation expressions
        const expressions = {
            objective: objective,
            constraints: constraints,
            nutritionalValues: {}
        };
        
        // Create expressions for nutritional calculations
        for (const [nutrient, target] of Object.entries(targets)) {
            const nutritionExpr = this.buildNutritionalExpression(nutrient, ingredients);
            expressions.nutritionalValues[nutrient] = nutritionExpr;
        }
        
        this.formulationModels.set(id, {
            config,
            expressions,
            compiledModels: new Map()
        });
        
        // Compile all expressions
        for (const [key, expression] of Object.entries(expressions.nutritionalValues)) {
            const modelId = `formulation_${id}_${key}`;
            await this.platform.compile_model(
                modelId,
                `Formulation ${id}: ${key}`,
                expression,
                ingredients.map(ing => ing.name)
            );
            await this.platform.load_model(modelId);
            
            this.formulationModels.get(id).compiledModels.set(key, modelId);
        }
        
        this.log(`Created formulation model: ${id}`);
    }
    
    /**
     * Evaluate a formulation with given ingredient amounts
     * @param {string} id - Formulation ID
     * @param {Object} amounts - Ingredient amounts
     * @returns {Promise<Object>} Formulation results
     */
    async evaluateFormulation(id, amounts) {
        const formulation = this.formulationModels.get(id);
        if (!formulation) {
            throw new Error(`Formulation '${id}' not found`);
        }
        
        const ingredientNames = Object.keys(amounts);
        const ingredientValues = Object.values(amounts);
        
        const results = {
            nutritionalValues: {},
            cost: 0,
            constraintsSatisfied: true
        };
        
        // Evaluate nutritional values
        for (const [nutrient, modelId] of formulation.compiledModels) {
            try {
                const value = await this.platform.execute_model(modelId, ingredientValues);
                results.nutritionalValues[nutrient] = value;
            } catch (error) {
                this.log(`Error evaluating ${nutrient}: ${error.message}`);
                results.nutritionalValues[nutrient] = NaN;
            }
        }
        
        // Calculate total cost (if cost data available)
        for (const [ingredient, amount] of Object.entries(amounts)) {
            const ingredientData = this.ingredientDatabase.get(ingredient);
            if (ingredientData && ingredientData.cost) {
                results.cost += amount * ingredientData.cost;
            }
        }
        
        return results;
    }
    
    // Private helper method
    buildNutritionalExpression(nutrient, ingredients) {
        // Build expression: sum(ingredient_amount * nutrient_content)
        const terms = ingredients.map(ing => {
            const ingredientData = this.ingredientDatabase.get(ing.name);
            const nutrientContent = ingredientData?.[nutrient] || 0;
            return `${ing.name} * ${nutrientContent}`;
        });
        
        return terms.join(' + ');
    }
}

/**
 * Usage Examples
 */
async function demonstrateUsage() {
    console.log('=== Basic Math Compiler Usage ===');
    
    // Basic usage
    const compiler = new AdvancedMathCompiler({
        enableLogging: true,
        enableCache: true
    });
    
    // Simple expression evaluation
    let result = await compiler.evaluate('x^2 + y^2', { x: 3, y: 4 });
    console.log('Distance calculation:', result); // Should be 25
    
    // Batch evaluation
    const expressions = [
        'sin(x) + cos(y)',
        'sqrt(x^2 + y^2)',
        'exp(-x) * log(y + 1)'
    ];
    
    const results = await compiler.evaluateBatch(expressions, { x: 1, y: 2 });
    console.log('Batch results:', results);
    
    // Performance benchmark
    const benchmark = await compiler.benchmark(
        'sin(x) * cos(y) + sqrt(x^2 + y^2)',
        { x: 1.5, y: 2.5 },
        5000
    );
    console.log('Benchmark results:', benchmark);
    
    console.log('=== Real-time Evaluator Usage ===');
    
    // Real-time evaluator
    const realTimeEval = new RealTimeMathEvaluator();
    
    await realTimeEval.registerExpression(
        'distance', 
        'sqrt((x2-x1)^2 + (y2-y1)^2)',
        ['x1', 'y1', 'x2', 'y2']
    );
    
    // Simulate real-time updates
    for (let i = 0; i < 10; i++) {
        const distance = await realTimeEval.evaluateById(
            'distance',
            [0, 0, i, i]
        );
        console.log(`Frame ${i}: Distance = ${distance.toFixed(2)}`);
    }
    
    console.log('=== Formulation Compiler Usage ===');
    
    // Formulation compiler
    const formulationCompiler = new FormulationMathCompiler();
    
    // Register ingredients
    formulationCompiler.registerIngredient('corn', {
        protein: 8.5,
        fat: 3.8,
        fiber: 2.3,
        cost: 0.25
    });
    
    formulationCompiler.registerIngredient('soybean_meal', {
        protein: 44.0,
        fat: 1.5,
        fiber: 7.0,
        cost: 0.45
    });
    
    // Create formulation model
    await formulationCompiler.createFormulation('pig_feed', {
        objective: 'minimize cost',
        ingredients: [
            { name: 'corn' },
            { name: 'soybean_meal' }
        ],
        targets: {
            protein: 18.0,
            fat: 5.0
        },
        constraints: [
            'corn + soybean_meal <= 100',
            'corn >= 40',
            'soybean_meal >= 10'
        ]
    });
    
    // Evaluate formulation
    const formulationResult = await formulationCompiler.evaluateFormulation(
        'pig_feed',
        { corn: 70, soybean_meal: 25 }
    );
    
    console.log('Formulation results:', formulationResult);
    
    // Get comprehensive statistics
    const stats = await compiler.getStatistics();
    console.log('Compiler statistics:', stats);
}

// Export classes for use in other modules
export {
    AdvancedMathCompiler,
    RealTimeMathEvaluator,
    FormulationMathCompiler,
    demonstrateUsage
};

// Auto-run demonstration if this file is executed directly
if (typeof window !== 'undefined' && window.document) {
    // Browser environment
    window.addEventListener('DOMContentLoaded', () => {
        console.log('Math Compiler Integration Example loaded');
        // Uncomment to run demonstration:
        // demonstrateUsage().catch(console.error);
    });
} else if (typeof process !== 'undefined' && process.argv) {
    // Node.js environment
    if (process.argv[2] === '--demo') {
        demonstrateUsage().catch(console.error);
    }
}
