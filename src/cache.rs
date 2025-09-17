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
