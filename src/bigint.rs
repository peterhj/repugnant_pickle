/* Derived from the implementation in libcore.

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE. */

use std::fmt;

pub struct BigInt {
    bytesize:   usize,
    base:       Vec<u64>,
    neg:        bool,
}

impl BigInt {
    pub fn from_le_bytes(v: &[u8]) -> BigInt {
        let bytesize = v.len();
        let mut base = Vec::new();
        let word_lo = v.len() / 8;
        let mut t: [u8; 8] = [0; 8];
        for k in 0 .. word_lo {
            t.copy_from_slice(&v[k * 8 .. (k + 1) * 8]);
            let u = u64::from_le_bytes(t);
            base.push(u);
        }
        let align_lo = word_lo * 8;
        if align_lo < bytesize {
            let mut tmpu: u64 = 0;
            for p in align_lo .. bytesize {
                let offset = p - align_lo;
                tmpu |= (v[p] as u64) << (offset * 8);
            }
            base.push(tmpu);
        }
        BigInt { bytesize, base, neg: false }
    }

    pub fn push_le_byte(&mut self, v: u8) {
        let offset = self.bytesize % 8;
        if offset == 0 {
            self.base.push(v as u64);
        } else {
            let word_lo = self.bytesize / 8;
            let mut tmpu = self.base[word_lo];
            tmpu |= (v as u64) << (offset * 8);
            self.base[word_lo] = tmpu;
        }
        self.bytesize += 1;
    }

    pub fn set_sign(&mut self, neg: bool) {
        self.neg = neg;
    }

    pub fn try_cast_to_i64(&self) -> Result<i64, ()> {
        if self.bytesize <= 7 {
            let u = self.base[0] as i64;
            let mut i = u as i64;
            assert!(i >= 0);
            assert!(i <= 0x00ff_ffff_ffff_ffff);
            if self.neg {
                i = -i;
            }
            return Ok(i);
        }
        let last_byte = (self.base[0] >> 56) as u8;
        let ret;
        if last_byte <= 0x7f {
            let u = self.base[0];
            let mut i = u as i64;
            if self.neg {
                i = -i;
            }
            ret = i;
        } else if self.neg && last_byte == 0x80 {
            let u = self.base[0];
            ret = u as i64;
        } else {
            return Err(());
        }
        let word_lo = self.bytesize / 8;
        for k in 1 .. word_lo {
            let u = self.base[k];
            if u != 0 {
                return Err(());
            }
        }
        let align_lo = word_lo * 8;
        if align_lo < self.bytesize {
            let last_word = self.base[word_lo];
            for p in align_lo .. self.bytesize {
                let offset = p - align_lo;
                let byte = (last_word >> (offset * 8)) as u8;
                if byte != 0 {
                    return Err(());
                }
            }
        }
        Ok(ret)
    }
}

impl PartialEq for BigInt {
    fn eq(&self, other: &BigInt) -> bool {
        if self.neg ^ other.neg {
            return false;
        }
        if self.bytesize != other.bytesize {
            return false;
        }
        let word_lo = self.bytesize / 8;
        for k in 0 .. word_lo {
            let u1 = self.base[k];
            let u2 = other.base[k];
            if u1 != u2 {
                return false;
            }
        }
        let align_lo = word_lo * 8;
        if align_lo < self.bytesize {
            let u1 = self.base[word_lo];
            let u2 = other.base[word_lo];
            for p in align_lo .. self.bytesize {
                let offset = p - align_lo;
                let byte1 = (u1 >> (offset * 8)) as u8;
                let byte2 = (u2 >> (offset * 8)) as u8;
                if byte1 != byte2 {
                    return false;
                }
            }
        }
        true
    }
}

impl Clone for BigInt {
    fn clone(&self) -> BigInt {
        BigInt { bytesize: self.bytesize, base: self.base.clone(), neg: self.neg }
    }
}

impl fmt::Debug for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sz = if self.bytesize < 8 { 1 } else { (self.bytesize + 8 - 1) / 8 };
        if self.neg {
            write!(f, "-")?;
        }
        write!(f, "{:#x}", self.base[sz - 1])?;
        for &v in self.base[..sz - 1].iter().rev() {
            write!(f, "_{:016x}", v)?;
        }
        Ok(())
    }
}
