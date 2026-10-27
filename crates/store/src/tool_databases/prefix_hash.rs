use blake3::hazmat::{HasherExt, Mode, merge_subtrees_non_root, merge_subtrees_root};
use serde::{Deserialize, Serialize};

const BLOCK_BYTES: usize = 64 * 1024;

/// The completed subtrees and final subtree of an ordinary BLAKE3 hash.
/// Keeping the final subtree allows appends without reading earlier bytes.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(super) struct PrefixHash {
    chunks: u64,
    stack: Vec<[u8; 32]>,
    tail: Vec<u8>,
}

impl PrefixHash {
    pub fn len(&self) -> u64 {
        self.chunks * BLOCK_BYTES as u64 + self.tail.len() as u64
    }

    pub fn valid(&self) -> bool {
        self.chunks
            .checked_mul(BLOCK_BYTES as u64)
            .and_then(|length| length.checked_add(self.tail.len() as u64))
            .is_some()
            && self.tail.len() <= BLOCK_BYTES
            && self.stack.len() == self.chunks.count_ones() as usize
            && (self.chunks == 0 || !self.tail.is_empty())
    }

    pub fn update(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            if self.tail.len() == BLOCK_BYTES {
                let mut hash = blake3::Hasher::new();
                hash.set_input_offset(self.chunks * BLOCK_BYTES as u64)
                    .update(&self.tail);
                let mut node = hash.finalize_non_root();
                self.chunks += 1;
                let mut width = self.chunks;
                while width & 1 == 0 {
                    node = merge_subtrees_non_root(
                        &self.stack.pop().expect("left subtree"),
                        &node,
                        Mode::Hash,
                    );
                    width >>= 1;
                }
                self.stack.push(node);
                self.tail.clear();
            }
            let n = bytes.len().min(BLOCK_BYTES - self.tail.len());
            self.tail.extend_from_slice(&bytes[..n]);
            bytes = &bytes[n..];
        }
    }

    pub fn finalize(&self) -> blake3::Hash {
        if self.chunks == 0 {
            return blake3::hash(&self.tail);
        }
        let mut hash = blake3::Hasher::new();
        hash.set_input_offset(self.chunks * BLOCK_BYTES as u64)
            .update(&self.tail);
        let mut node = hash.finalize_non_root();
        for (i, left) in self.stack.iter().enumerate().rev() {
            if i == 0 {
                return merge_subtrees_root(left, &node, Mode::Hash);
            }
            node = merge_subtrees_non_root(left, &node, Mode::Hash);
        }
        unreachable!("a nonempty prefix has a left subtree")
    }
}

#[cfg(test)]
mod tests;
