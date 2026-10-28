use super::{Observer, Tree};
use std::collections::BTreeSet;
use std::io::{self, BufRead, BufReader, Read};

fn invalid() -> serde_json::Error {
    <serde_json::Error as serde::de::Error>::custom("invalid JSON record")
}

pub(super) fn read(reader: impl Read, names: &BTreeSet<String>) -> Result<Tree, serde_json::Error> {
    let mut parser = Parser {
        reader: BufReader::new(reader),
        parents: 0,
    };
    let tree = parser.value(names)?;
    if parser.space()?.is_some() {
        return Err(invalid());
    }
    Ok(tree)
}

pub(super) fn array(
    reader: impl BufRead,
    names: &BTreeSet<String>,
) -> Result<Observer, serde_json::Error> {
    let mut parser = Parser { reader, parents: 0 };
    if parser.space()? != Some(b'[') {
        return Err(invalid());
    }
    parser.byte()?;
    let mut whole = Observer::new(names.clone());
    if parser.space()? != Some(b']') {
        loop {
            let tree = parser.value(names)?;
            whole.root.merge(tree, names);
            whole.records += 1;
            match parser.space()? {
                Some(b',') => {
                    parser.byte()?;
                }
                Some(b']') => break,
                _ => return Err(invalid()),
            }
        }
    }
    parser.byte()?;
    if parser.space()?.is_some() {
        return Err(invalid());
    }
    Ok(whole)
}

struct Parser<R> {
    reader: R,
    parents: u64,
}

struct Frame {
    node: usize,
    parent: u64,
    array: bool,
    first: bool,
}

impl<R: BufRead> Parser<R> {
    fn peek(&mut self) -> Result<Option<u8>, serde_json::Error> {
        Ok(self
            .reader
            .fill_buf()
            .map_err(serde_json::Error::io)?
            .first()
            .copied())
    }

    fn byte(&mut self) -> Result<u8, serde_json::Error> {
        let byte = self.peek()?.ok_or_else(invalid)?;
        self.reader.consume(1);
        Ok(byte)
    }

    fn skip_while(&mut self, accepts: fn(u8) -> bool) -> Result<bool, serde_json::Error> {
        let mut any = false;
        loop {
            let bytes = self.reader.fill_buf().map_err(serde_json::Error::io)?;
            let count = bytes.iter().take_while(|byte| accepts(**byte)).count();
            if count == 0 {
                return Ok(any);
            }
            any = true;
            let ended = count < bytes.len();
            self.reader.consume(count);
            if ended {
                return Ok(any);
            }
        }
    }

    fn space(&mut self) -> Result<Option<u8>, serde_json::Error> {
        self.skip_while(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))?;
        self.peek()
    }

    fn expect(&mut self, expected: u8) -> Result<(), serde_json::Error> {
        if self.byte()? == expected {
            Ok(())
        } else {
            Err(invalid())
        }
    }

    fn value(&mut self, names: &BTreeSet<String>) -> Result<Tree, serde_json::Error> {
        let mut tree = Tree::default();
        let root = tree.root();
        tree.nodes[root].present = 1;
        let mut frames = Vec::<Frame>::new();
        let mut next = Some(root);
        loop {
            if let Some(node) = next.take() {
                let byte = self.space()?.ok_or_else(invalid)?;
                let kind = match byte {
                    b'{' | b'[' => {
                        self.byte()?;
                        self.parents += 1;
                        frames.push(Frame {
                            node,
                            parent: self.parents,
                            array: byte == b'[',
                            first: true,
                        });
                        if byte == b'[' { "array" } else { "object" }
                    }
                    b'"' => {
                        self.string(false)?;
                        "string"
                    }
                    b't' => {
                        self.literal(b"true")?;
                        "boolean"
                    }
                    b'f' => {
                        self.literal(b"false")?;
                        "boolean"
                    }
                    b'n' => {
                        self.literal(b"null")?;
                        "null"
                    }
                    b'-' | b'0'..=b'9' => self.number()?,
                    _ => return Err(invalid()),
                };
                tree.kind(node, kind);
            }
            let Some(frame) = frames.last_mut() else {
                break;
            };
            let close = if frame.array { b']' } else { b'}' };
            let byte = self.space()?.ok_or_else(invalid)?;
            if byte == close {
                self.byte()?;
                frames.pop();
                continue;
            }
            if !frame.first {
                self.expect(b',')?;
                if self.space()? == Some(close) {
                    return Err(invalid());
                }
            }
            frame.first = false;
            let child = if frame.array {
                let index = tree.child(frame.node, "[]".into());
                if tree.nodes[index].last_parent != frame.parent {
                    tree.nodes[index].present += 1;
                    tree.nodes[index].last_parent = frame.parent;
                }
                index
            } else {
                let name = self.string(true)?;
                self.space()?;
                self.expect(b':')?;
                tree.member(frame.node, &name, frame.parent, names)
            };
            next = Some(child);
        }
        Ok(tree)
    }

    fn literal(&mut self, text: &[u8]) -> Result<(), serde_json::Error> {
        for byte in text {
            self.expect(*byte)?;
        }
        Ok(())
    }

    fn digits(&mut self) -> Result<(), serde_json::Error> {
        if self.skip_while(|byte| byte.is_ascii_digit())? {
            Ok(())
        } else {
            Err(invalid())
        }
    }

    fn number(&mut self) -> Result<&'static str, serde_json::Error> {
        if self.peek()? == Some(b'-') {
            self.byte()?;
        }
        if self.peek()? == Some(b'0') {
            self.byte()?;
        } else {
            self.digits()?;
        }
        let mut kind = "integer";
        if self.peek()? == Some(b'.') {
            self.byte()?;
            self.digits()?;
            kind = "number";
        }
        if self.peek()?.is_some_and(|c| matches!(c, b'e' | b'E')) {
            self.byte()?;
            if self.peek()?.is_some_and(|c| matches!(c, b'+' | b'-')) {
                self.byte()?;
            }
            self.digits()?;
            kind = "number";
        }
        Ok(kind)
    }

    fn hex(&mut self) -> Result<u16, serde_json::Error> {
        let mut number = 0;
        for _ in 0..4 {
            let digit = (self.byte()? as char).to_digit(16).ok_or_else(invalid)?;
            number = number * 16 + digit as u16;
        }
        Ok(number)
    }

    // Values are validated and discarded. Only field names need decoded storage.
    fn string(&mut self, keep: bool) -> Result<String, serde_json::Error> {
        self.expect(b'"')?;
        let mut text = String::new();
        loop {
            // Validate whole UTF-8 runs; only a split character needs byte reads.
            let bytes = self.reader.fill_buf().map_err(serde_json::Error::io)?;
            let count = bytes
                .iter()
                .take_while(|c| **c >= 0x20 && **c != b'"' && **c != b'\\')
                .count();
            if count > 0 {
                let valid = match std::str::from_utf8(&bytes[..count]) {
                    Ok(text) => text,
                    Err(error) if error.error_len().is_none() => {
                        std::str::from_utf8(&bytes[..error.valid_up_to()]).expect("valid prefix")
                    }
                    Err(_) => return Err(invalid()),
                };
                if !valid.is_empty() {
                    if keep {
                        text.push_str(valid);
                    }
                    let length = valid.len();
                    self.reader.consume(length);
                    continue;
                }
            }
            let byte = self.byte()?;
            let character = match byte {
                b'"' => return Ok(text),
                b'\\' => match self.byte()? {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let first = self.hex()?;
                        let code = if (0xd800..=0xdbff).contains(&first) {
                            self.expect(b'\\')?;
                            self.expect(b'u')?;
                            let second = self.hex()?;
                            if !(0xdc00..=0xdfff).contains(&second) {
                                return Err(invalid());
                            }
                            0x10000 + ((first as u32 - 0xd800) << 10) + second as u32 - 0xdc00
                        } else {
                            first as u32
                        };
                        char::from_u32(code).ok_or_else(invalid)?
                    }
                    _ => return Err(invalid()),
                },
                0..=0x1f => return Err(invalid()),
                _ => {
                    let length = match byte {
                        0xc2..=0xdf => 2,
                        0xe0..=0xef => 3,
                        0xf0..=0xf4 => 4,
                        _ => return Err(invalid()),
                    };
                    let mut bytes = [byte, 0, 0, 0];
                    for item in bytes.iter_mut().take(length).skip(1) {
                        *item = self.byte()?;
                    }
                    std::str::from_utf8(&bytes[..length])
                        .map_err(|_| invalid())?
                        .chars()
                        .next()
                        .expect("character")
                }
            };
            if keep {
                text.push(character);
            }
        }
    }
}

pub(super) struct Line<'a, R> {
    pub reader: &'a mut R,
    pub ended: bool,
    pub nonwhite: bool,
}
impl<R: BufRead> Read for Line<'_, R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.ended || out.is_empty() {
            return Ok(0);
        }
        let bytes = self.reader.fill_buf()?;
        let length = bytes
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |n| n + 1)
            .min(out.len());
        out[..length].copy_from_slice(&bytes[..length]);
        self.nonwhite |= bytes[..length]
            .iter()
            .any(|byte| !byte.is_ascii_whitespace());
        self.ended = length > 0 && bytes[length - 1] == b'\n';
        self.reader.consume(length);
        Ok(length)
    }
}
