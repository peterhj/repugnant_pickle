//! This is simple support for PyTorch files (.pth)
//! It really won't be able to handle anything fancy.
//! If your Torch file is just a flat dict without any
//! exotic types, you have a chance of this working.
//!
//! Example result:
//!
//! ```plaintext
//! RepugnantTorchTensors(
//!    [
//!        RepugnantTorchTensor {
//!            name: "emb.weight",
//!            device: "cuda:0",
//!            tensor_type: BFloat16,
//!            storage: "archive/data/0",
//!            storage_len: 430348288,
//!            storage_offset: 327378944,
//!            absolute_offset: 327445248,
//!            shape: [1024, 50277],
//!            stride: [1, 1024],
//!            requires_grad: false,
//!        },
//!        RepugnantTorchTensor {
//!            name: "blocks.0.ln1.weight",
//!            device: "cuda:0",
//!            tensor_type: BFloat16,
//!            storage: "archive/data/0",
//!            storage_len: 430348288,
//!            storage_offset: 13639680,
//!            absolute_offset: 13705984,
//!            shape: [1024],
//!            stride: [1],
//!            requires_grad: false,
//!        },
//!    ]
//! ```
//!
//! If you mmap the whole file, you can access the tensors
//! starting at the absolute offset. You will need to calculate
//! the length from the shape and type.
//! Alternatively, you can open the Torch file as a ZIP and
//! read it the ld fashioned way using `storage` as the ZIP
//! member filename.

use crate::{ops::PickleOp, *};

use anyhow::{anyhow, bail, ensure, Ok, Result};
use smol_str::{SmolStr};

use std::{borrow::Cow, fs::File, io::Read, path::Path, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    Float64,
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Int16,
    Int8,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    //Unknown(String),
}

impl TensorType {
    /// Get the item size for this tensor type. However,
    /// the type of Unknown tensor types is... well,
    /// unknown. So you get 0 back there.
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorType::Float64 => 8,
            TensorType::Float32 => 4,
            TensorType::Float16 => 2,
            TensorType::BFloat16 => 2,
            TensorType::Int64 => 8,
            TensorType::Int32 => 4,
            TensorType::Int16 => 2,
            TensorType::Int8 => 1,
            TensorType::UInt64 => 8,
            TensorType::UInt32 => 4,
            TensorType::UInt16 => 2,
            TensorType::UInt8 => 1,
            //TensorType::Unknown(_) => 0,
        }
    }
}

impl FromStr for TensorType {
    //type Err = std::convert::Infallible;
    type Err = SmolStr;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.strip_suffix("Storage").unwrap_or(s).to_ascii_lowercase();
        std::result::Result::Ok(match s.as_str() {
            "float64" | "double" => Self::Float64,
            "float32" | "float" => Self::Float32,
            "float16" | "half" => Self::Float16,
            "bfloat16" => Self::BFloat16,
            "int64" | "long" => Self::Int64,
            "int32" | "int" => Self::Int32,
            "int16" | "short" => Self::Int16,
            "int8" | "char" => Self::Int8,
            "uint64" => Self::UInt64,
            "uint32" => Self::UInt32,
            "uint16" => Self::UInt16,
            "uint8" | "byte" => Self::UInt8,
            //_ => Self::Unknown(s),
            _ => return Err(s.into())
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepugnantTorchTensor {
    /// Tensor name.
    pub name: SmolStr,

    /// Device
    pub device: SmolStr,

    /// Type of tensor.
    pub tensor_type: std::result::Result<TensorType, SmolStr>,

    /// The filename in the ZIP which has storage for this tensor.
    pub storage: SmolStr,

    /// Total length (in bytes) for the entire storage item.
    /// Note that multiple tensors can point to different ranges
    /// of the item. Or maybe even the same ranges.
    pub storage_len: u64,

    /// Offset into the storage file where this tensor's data starts.
    /// Torch files don't have ZIP compression enabled so you can
    /// use this for mmaping the whole file and extracting the tensor data.
    /// However bear in mind it won't necessarily be aligned.
    pub storage_offset: u64,

    /// Absolute offset into the Torch (zip) file.
    pub absolute_offset: u64,

    /// The tensor shape (dimensions).
    pub shape: Vec<i64>,

    /// The tensor stride.
    pub stride: Vec<i64>,

    /// Whether the tensor requires gradients enabled.
    pub requires_grad: bool,
}

pub struct RepugnantTorchTensorsIter<'a> {
    index: usize,
    zipfile: &'a mut zip::ZipArchive<File>,
    tensors: &'a mut [RepugnantTorchTensor],
}

impl<'a> RepugnantTorchTensorsIter<'a> {
    // FIXME FIXME: anyhow/result.
    //pub fn next_tensor(&'a mut self) -> Option<(&'a RepugnantTorchTensor, Result<zip::read::ZipFile<'a>>)> {}
    pub fn next_tensor(&'a mut self) -> Option<(&'a RepugnantTorchTensor, zip::read::ZipFile<'a>)> {
        if self.index >= self.tensors.len() {
            return None;
        }
        let idx = self.index;
        self.index += 1;
        // This actually shouldn't ever fail.
        let zf = self.zipfile.by_name(&self.tensors[idx].storage)
                             //.map_err(|_| anyhow!("Missing tensor storage in zip archive"));
                             .unwrap();
        // FIXME FIXME: before, was using anyhow ensure!.
        assert_eq!(
            zf.compression(), zip::CompressionMethod::STORE,
            "Can't handle compressed tensor files",
        );
        if self.tensors[idx].absolute_offset == 0 {
            let offs = self.tensors[idx].storage_offset;
            self.tensors[idx].absolute_offset = zf.data_start() + offs;
            assert!(self.tensors[idx].absolute_offset != 0);
        }
        Some((&self.tensors[idx], zf))
    }
}

pub struct RepugnantTorchFile {
    zipfile: zip::ZipArchive<File>,
    tensors: Vec<RepugnantTorchTensor>,
}

impl RepugnantTorchFile {
    pub fn open<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let mut zp = zip::ZipArchive::new(File::open(filename)?)?;

        let datafn = zp
            .file_names()
            .find(|s| s.ends_with("/data.pkl"))
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("Could not find data.pkl in archive"))?;
        let (pfx, _) = datafn.rsplit_once('/').unwrap();
        let mut zf = zp.by_name(&datafn)?;
        //println!("DEBUG: RepugnantTorchFile: compression={:?}", zf.compression());
        let mut buf = Vec::with_capacity(zf.size() as usize);
        let _ = zf.read_to_end(&mut buf)?;
        drop(zf);
        let (_remain, ops) = parse_ops::<nom::error::VerboseError<&[u8]>>(&buf)
            .map_err(|e| anyhow!("Parse error: {:?}", e))?;

        // Why _wouldn't_ there be random garbage left after parsing the pickle?
        // ensure!(!remain.is_empty(), "Unexpected remaining data in pickle");

        let (vals, _memo) = evaluate(&ops, true)?;
        let vals = vals.as_slice();
        //let n_vals = vals.len();
        //println!("DEBUG: RepugnantTorchFile: vals.len={}", n_vals);
        //println!("DEBUG: RepugnantTorchFile: vals={:?}", vals);
        /*let val = match (&vals, n_vals) {
            (&[Value::Build(a, _), ..], _) => a.as_ref(),
            (&[Value::Seq(..), ..], 1) => &vals[0],
            _ => bail!("Unexpected toplevel type"),
        };*/
        let val = match &vals {
            &[Value::Build(a, _), ..] => a.as_ref(),
            &[Value::Seq(..)] => &vals[0],
            _ => bail!("Unexpected toplevel type"),
        };
        // Presumably this is usually going to be an OrderedDict, but maybe
        // it can also be a plain old Dict.
        let val = match val {
            Value::Global(g, seq) => match g.as_ref() {
                // Dereffing both the Box and Cow here.
                Value::Raw(rv) if **rv == PickleOp::GLOBAL("collections", "OrderedDict") => {
                    match seq.as_slice() {
                        [_, Value::Seq(SequenceType::Tuple, seq2), ..] => seq2,
                        _ => bail!("Unexpected value in collections.OrderedDict"),
                    }
                }
                _ => bail!("Unexpected type in toplevel Global"),
            },
            Value::Seq(SequenceType::Dict, seq) => seq,
            _ => bail!("Unexpected type in Build"),
        };
        //println!("DEBUG: RepugnantTorchFile: n_val={}", val.len());
        //println!("DEBUG: RepugnantTorchFile: val={:?}", &val);
        let mut tensors = Vec::with_capacity(16);
        for di in val.iter() {
            let (k, v) = match di {
                Value::Seq(SequenceType::Tuple, seq) if seq.len() == 2 => (&seq[0], &seq[1]),
                _ => bail!("Could not get key/value for dictionary item"),
            };
            let k = if let Value::String(s) = k {
                *s
            } else {
                bail!("Dictionary key is not a string");
            };
            let v = match v {
                Value::Global(g, seq)
                    if g.as_ref()
                        == &Value::Raw(Cow::Owned(PickleOp::GLOBAL(
                            "torch._utils",
                            "_rebuild_tensor_v2",
                        ))) =>
                {
                    seq
                }
                // It's possible to jam random values into the Dict, so
                // since it's not a tensor we just ignore it here.
                _ => continue,
            };
            // println!("\nKey: {k:?}\n{v:?}");

            let (pidval, offs, shape, stride, grad) = match v.as_slice() {
                [Value::Seq(SequenceType::Tuple, seq)] => match seq.as_slice() {
                    [Value::PersId(pidval), Value::Int(offs), Value::Seq(SequenceType::Tuple, shape), Value::Seq(SequenceType::Tuple, stride), Value::Bool(grad), ..] => {
                        (pidval.as_ref(), *offs as u64, shape, stride, *grad)
                    }
                    _ => bail!("Unexpected value in call to torch._utils._rebuild_tensor_v2"),
                },
                _ => bail!("Unexpected type in call to torch._utils._rebuild_tensor_v2"),
            };
            // println!("PID: {pidval:?}");
            let fixdim = |v: &[Value]| {
                v.iter()
                    .map(|x| match x {
                        Value::Int(n) => Ok(*n),
                        _ => bail!("Bad value for shape/stride item"),
                    })
                    .collect::<Result<Vec<_>>>()
            };
            let shape = fixdim(shape)?;
            let stride = fixdim(stride)?;
            // println!("Tensor: shape={shape:?}, stride={stride:?}, offs={offs}, grad={grad:?}");
            let (stype, sfile, sdev, slen) = match pidval {
                Value::Seq(SequenceType::Tuple, seq) => match seq.as_slice() {
                    [Value::String("storage"), Value::Raw(op), Value::String(sfile), Value::String(sdev), Value::Int(slen)] => {
                        match &**op {
                            PickleOp::GLOBAL("torch", styp) if styp.ends_with("Storage") => {
                                (&styp[..styp.len() - 7], *sfile, *sdev, *slen as u64)
                            }
                            _ => bail!("Unexpected storage type part of persistant ID"),
                        }
                    }
                    _ => bail!("Unexpected sequence in persistant ID"),
                },
                _ => bail!("Unexpected value for persistant ID"),
            };
            let stype = TensorType::from_str(stype);
            let sfile = format!("{pfx}/data/{sfile}");

            // println!("PID: file={sfile}, len={slen}, type={stype:?}, dev={sdev}");

            let offs = offs * stype.as_ref().map(|t| t.size_bytes() as u64).unwrap_or(0);
            tensors.push(RepugnantTorchTensor{
                name: k.into(),
                device: sdev.into(),
                tensor_type: stype,
                storage: sfile.into(),
                storage_len: slen,
                storage_offset: offs,
                /*absolute_offset: zf.data_start() + offs,*/
                absolute_offset: 0,
                shape,
                stride,
                requires_grad: grad,
            })
        }
        Ok(Self{zipfile: zp, tensors})
    }

    pub fn tensors(&self) -> &[RepugnantTorchTensor] {
        &self.tensors
    }

    pub fn iter_tensors_data<'a>(&'a mut self) -> RepugnantTorchTensorsIter<'a> {
        RepugnantTorchTensorsIter{
            index: 0,
            zipfile: &mut self.zipfile,
            tensors: &mut self.tensors,
        }
    }
}
