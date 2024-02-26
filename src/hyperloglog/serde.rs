use super::HyperLogLog;
use serde::de::{self, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

impl<T, B> Serialize for HyperLogLog<T, B>
where
    T: Hash + ?Sized,
    B: BuildHasher + Clone + Eq + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("HyperLogLog", 3)?;
        state.serialize_field("registers", &self.registers)?;
        state.serialize_field("b", &self.b)?;
        state.serialize_field("buildhasher", &self.buildhasher)?;
        state.end()
    }
}

impl<'de, T, B> Deserialize<'de> for HyperLogLog<T, B>
where
    T: Hash + ?Sized,
    B: BuildHasher + Clone + Eq + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum Field {
            Registers,
            B,
            Buildhasher,
        }
        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;
                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                        formatter.write_str("`registers` or `b` or `buildhasher")
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: de::Error,
                    {
                        match v {
                            "registers" => Ok(Field::Registers),
                            "b" => Ok(Field::B),
                            "buildhasher" => Ok(Field::Buildhasher),
                            _ => Err(de::Error::unknown_field(v, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }
        struct HyperLogLogVisitor<T, B>
        where
            T: Hash + ?Sized,
            B: BuildHasher + Clone + Eq,
        {
            _t: PhantomData<T>,
            _b: PhantomData<B>,
        }
        impl<T, B> HyperLogLogVisitor<T, B>
        where
            T: Hash + ?Sized,
            B: BuildHasher + Clone + Eq,
        {
            fn new() -> Self {
                Self {
                    _t: PhantomData,
                    _b: PhantomData,
                }
            }
        }
        impl<'de, T, B> Visitor<'de> for HyperLogLogVisitor<T, B>
        where
            T: Hash + ?Sized,
            B: BuildHasher + Clone + Eq + Deserialize<'de>,
        {
            type Value = HyperLogLog<T, B>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("struct HyperLogLog")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut registers = None;
                let mut b = None;
                let mut buildhasher = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Registers => {
                            if registers.is_some() {
                                return Err(de::Error::duplicate_field("registers"));
                            }
                            registers = Some(map.next_value()?);
                        }
                        Field::B => {
                            if b.is_some() {
                                return Err(de::Error::duplicate_field("b"));
                            }
                            b = Some(map.next_value()?);
                        }
                        Field::Buildhasher => {
                            if buildhasher.is_some() {
                                return Err(de::Error::duplicate_field("buildhasher"));
                            }
                            buildhasher = Some(map.next_value()?);
                        }
                    }
                }
                let registers = registers.ok_or_else(|| de::Error::missing_field("registers"))?;
                let b = b.ok_or_else(|| de::Error::missing_field("b"))?;
                let buildhasher =
                    buildhasher.ok_or_else(|| de::Error::missing_field("buildhasher"))?;
                Ok(HyperLogLog {
                    registers,
                    b,
                    buildhasher,
                    phantom: PhantomData,
                })
            }
        }
        const FIELDS: &[&str] = &["registers", "b", "buildhasher"];
        deserializer.deserialize_struct("HyperLogLog", FIELDS, HyperLogLogVisitor::<T, B>::new())
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, Hasher};

    use serde::{Deserialize, Serialize};

    use crate::hyperloglog::HyperLogLog;

    #[test]
    fn serde() {
        #[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
        struct MyHasher {
            state: u64,
        }
        impl Hasher for MyHasher {
            fn finish(&self) -> u64 {
                self.state
            }

            fn write(&mut self, bytes: &[u8]) {
                let _ = bytes;
            }
        }
        impl BuildHasher for MyHasher {
            type Hasher = Self;
            fn build_hasher(&self) -> Self::Hasher {
                Self { state: 4 }
            }
        }
        let hasher = MyHasher { state: 4 };
        // Construct a HLL
        let mut hll = HyperLogLog::with_hash(4, hasher.clone());
        hll.add("abc");
        // Serialize to JSON
        let json = serde_json::to_string(&hll).expect("can serialize to json");
        // Deserialize back to HLL
        let mut de_hll = serde_json::from_str(&json).expect("can deserialize from json");
        // Check they're the same
        assert_eq!(hll, de_hll);
        // Add the same string again to check if the hasher is reconstructed correctly
        de_hll.add("abc");
        assert_eq!(hll, de_hll);
    }
}
