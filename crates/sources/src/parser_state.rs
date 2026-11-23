//! History-sized parser lookups use the supplied context, not carried JSON maps.
use polylogue_model::parser_context::{Handle, Key, Value};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct Map<T> {
    #[serde(skip)]
    context: Handle,
    #[serde(skip)]
    namespace: String,
    #[serde(skip)]
    value: PhantomData<T>,
}

impl<T> Default for Map<T> {
    fn default() -> Self {
        Self {
            context: Default::default(),
            namespace: String::new(),
            value: PhantomData,
        }
    }
}
impl<T> Clone for Map<T> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            namespace: self.namespace.clone(),
            value: PhantomData,
        }
    }
}
impl<T> Map<T> {
    pub(crate) fn bind(&mut self, context: Handle, namespace: String) {
        self.context = context;
        self.namespace = namespace;
    }
    pub(crate) fn is_bound(&self) -> bool {
        !self.namespace.is_empty()
    }
    fn key(&self, key: &str) -> Key {
        Key::State {
            namespace: self.namespace.clone(),
            key: key.into(),
        }
    }
    pub(crate) fn clear(&mut self) {
        self.context.clear_state(&self.namespace);
    }
}
impl<T: Serialize + DeserializeOwned> Map<T> {
    pub(crate) fn get(&self, key: &str) -> Option<T> {
        match self.context.get(&self.key(key)) {
            Some(Value::State { value }) if !value.is_null() => match serde_json::from_value(value)
            {
                Ok(value) => Some(value),
                Err(error) => {
                    self.context.fail(error.to_string());
                    None
                }
            },
            Some(Value::State { .. }) | None => None,
            Some(_) => {
                self.context.fail("parser state has another value kind");
                None
            }
        }
    }
    pub(crate) fn insert(&self, key: String, value: T) {
        match serde_json::to_value(value) {
            Ok(value) => self.context.put(self.key(&key), Value::State { value }),
            Err(error) => self.context.fail(error.to_string()),
        }
    }
    pub(crate) fn get_mut(&self, key: &str) -> Option<Entry<T>> {
        self.get(key).map(|value| Entry {
            map: self.clone(),
            key: key.into(),
            value,
        })
    }
    pub(crate) fn get_or_insert(&self, key: String, make: impl FnOnce() -> T) -> Entry<T> {
        Entry {
            value: self.get(&key).unwrap_or_else(make),
            key,
            map: self.clone(),
        }
    }
    pub(crate) fn remove(&self, key: &str) -> Option<T> {
        let value = self.get(key);
        self.context.put(
            self.key(key),
            Value::State {
                value: serde_json::Value::Null,
            },
        );
        value
    }
}

pub(crate) struct Entry<T: Serialize + DeserializeOwned> {
    map: Map<T>,
    key: String,
    value: T,
}
impl<T: Serialize + DeserializeOwned> Deref for Entry<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
impl<T: Serialize + DeserializeOwned> DerefMut for Entry<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
impl<T: Serialize + DeserializeOwned> Drop for Entry<T> {
    fn drop(&mut self) {
        match serde_json::to_value(&self.value) {
            Ok(value) => self
                .map
                .context
                .put(self.map.key(&self.key), Value::State { value }),
            Err(error) => self.map.context.fail(error.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct Set<T> {
    values: Map<bool>,
    #[serde(skip)]
    value: PhantomData<T>,
}
impl<T> Default for Set<T> {
    fn default() -> Self {
        Self {
            values: Default::default(),
            value: PhantomData,
        }
    }
}
impl<T: Serialize> Set<T> {
    pub(crate) fn bind(&mut self, context: Handle, namespace: String) {
        self.values.bind(context, namespace);
    }
    pub(crate) fn insert(&self, value: T) -> bool {
        let key = match serde_json::to_string(&value) {
            Ok(key) => key,
            Err(error) => {
                self.values.context.fail(error.to_string());
                return false;
            }
        };
        if self.values.get(&key) == Some(true) {
            false
        } else {
            self.values.insert(key, true);
            true
        }
    }
}

#[cfg(test)]
mod tests;
