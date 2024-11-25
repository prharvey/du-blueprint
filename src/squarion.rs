use core::str;
use std::array::TryFromSliceError;
use std::collections::BTreeMap;
use std::ffi::c_char;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::io::{Error, Read};
use std::ops::Range;
use std::str::Utf8Error;

use parry3d_f64::math::{Point, Vector};
use rangemap::RangeMap;
use serde::ser::SerializeStruct;

#[derive(Debug)]
#[allow(dead_code)]
pub enum SerializeError {
    Internal(Error),
    BadData,
}

impl From<Error> for SerializeError {
    fn from(value: Error) -> Self {
        SerializeError::Internal(value)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum DeserializeError {
    Internal(Error),
    BadData,
    BadMagic(u32, u32),
    BadVersion(u32, u32),
}

fn assert_magic(actual: u32, expected: u32) -> Result<(), DeserializeError> {
    if actual != expected {
        Err(DeserializeError::BadMagic(actual, expected))
    } else {
        Ok(())
    }
}

fn assert_version(actual: u32, expected: u32) -> Result<(), DeserializeError> {
    if actual != expected {
        Err(DeserializeError::BadVersion(actual, expected))
    } else {
        Ok(())
    }
}

impl From<Error> for DeserializeError {
    fn from(value: Error) -> Self {
        DeserializeError::Internal(value)
    }
}

impl From<Utf8Error> for DeserializeError {
    fn from(_: Utf8Error) -> Self {
        DeserializeError::BadData
    }
}

impl From<TryFromSliceError> for DeserializeError {
    fn from(_: TryFromSliceError) -> Self {
        DeserializeError::BadData
    }
}

const COMPRESSED_MAGIC: u32 = 0xfb14b6f9;

pub trait Serialize {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError>;

    fn compress(&self) -> Result<Vec<u8>, SerializeError> {
        let mut result = Vec::new();
        COMPRESSED_MAGIC.serialize(&mut result)?;

        let mut uncompressed_data = Vec::new();
        self.serialize(&mut uncompressed_data)?;
        (uncompressed_data.len() as u64).serialize(&mut result)?;
        let mut compressed_data = vec![0; uncompressed_data.len() * 2];
        unsafe {
            let len = lz4::liblz4::LZ4_compress_default(
                uncompressed_data.as_ptr() as *const c_char,
                compressed_data.as_mut_ptr() as *mut c_char,
                uncompressed_data.len() as i32,
                compressed_data.len() as i32,
            );
            result.extend_from_slice(&compressed_data[0..len as usize]);
        }
        Ok(result)
    }
}

pub trait Deserialize
where
    Self: Sized,
{
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError>;

    fn decompress(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let magic = u32::from_le_bytes(bytes[..4].try_into()?);
        assert_magic(magic, COMPRESSED_MAGIC)?;
        let uncompressed_size = usize::from_le_bytes(bytes[4..12].try_into()?);
        let mut result = vec![0u8; uncompressed_size];
        assert!(bytes.len() >= 12);
        let compressed_size = bytes.len() - 12;
        unsafe {
            let decompressed = lz4::liblz4::LZ4_decompress_safe(
                bytes.as_ptr().add(12) as *const c_char,
                result.as_mut_ptr() as *mut c_char,
                compressed_size as i32,
                result.len() as i32,
            );
            if decompressed != result.len() as i32 {
                return Err(DeserializeError::BadData);
            }
        }
        let mut reader = result.as_slice();
        Self::deserialize(&mut reader)
    }
}

impl Serialize for u8 {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        writer.write_all(&[*self])?;
        Ok(())
    }
}

impl Deserialize for u8 {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut buf = [0];
        reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }
}

impl Serialize for u32 {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl Deserialize for u32 {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut buf = [0; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }
}

impl Serialize for i32 {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl Deserialize for i32 {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut buf = [0; 4];
        reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }
}

impl Serialize for u64 {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl Deserialize for u64 {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut buf = [0; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}

impl Serialize for i64 {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl Deserialize for i64 {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut buf = [0; 8];
        reader.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }
}

impl Serialize for f64 {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl Deserialize for f64 {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut buf = [0; 8];
        reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}

impl Serialize for Point<i32> {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        for v in self.iter() {
            v.serialize(writer)?;
        }
        Ok(())
    }
}

impl Deserialize for Point<i32> {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        Ok(Point::new(
            i32::deserialize(reader)?,
            i32::deserialize(reader)?,
            i32::deserialize(reader)?,
        ))
    }
}

impl Serialize for Point<f64> {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        for v in self.iter() {
            v.serialize(writer)?;
        }
        Ok(())
    }
}

impl Deserialize for Point<f64> {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        Ok(Point::new(
            f64::deserialize(reader)?,
            f64::deserialize(reader)?,
            f64::deserialize(reader)?,
        ))
    }
}

impl Serialize for Vector<i32> {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        for v in self.iter() {
            v.serialize(writer)?;
        }
        Ok(())
    }
}

impl Deserialize for Vector<i32> {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        Ok(Vector::new(
            i32::deserialize(reader)?,
            i32::deserialize(reader)?,
            i32::deserialize(reader)?,
        ))
    }
}

impl<T> Serialize for Option<T>
where
    T: Serialize,
{
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        match self {
            Some(v) => {
                1u8.serialize(writer)?;
                v.serialize(writer)
            }
            None => 0u8.serialize(writer),
        }
    }
}

impl<T> Deserialize for Option<T>
where
    T: Deserialize,
{
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        match u8::deserialize(reader)? {
            0 => Ok(None),
            _ => Ok(Some(T::deserialize(reader)?)),
        }
    }
}

impl<K, V> Serialize for BTreeMap<K, V>
where
    K: Serialize + Ord,
    V: Serialize,
{
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        (self.len() as u32).serialize(writer)?;
        for (k, v) in self {
            k.serialize(writer)?;
            v.serialize(writer)?;
        }
        Ok(())
    }
}

impl<K, V> Deserialize for BTreeMap<K, V>
where
    K: Deserialize + Ord,
    V: Deserialize,
{
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mut map = BTreeMap::new();
        let count = u32::deserialize(reader)?;
        for _ in 0..count {
            let key = K::deserialize(reader)?;
            let value = V::deserialize(reader)?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

impl<T> Serialize for [T; 3]
where
    T: Serialize,
{
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        for v in self {
            v.serialize(writer)?;
        }
        Ok(())
    }
}

impl<T> Deserialize for [T; 3]
where
    T: Deserialize,
{
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        Ok([
            T::deserialize(reader)?,
            T::deserialize(reader)?,
            T::deserialize(reader)?,
        ])
    }
}

fn serialize_rle(
    mut count: usize,
    value: &impl Serialize,
    writer: &mut impl Write,
) -> Result<(), SerializeError> {
    while count != 0 {
        value.serialize(writer)?;
        count -= 1;
        let more = count.min(255);
        (more as u8).serialize(writer)?;
        count -= more;
    }
    Ok(())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, serde::Serialize)]
pub struct VertexMaterial {
    pub material: u8,
}

impl VertexMaterial {
    pub fn new(material: u8) -> VertexMaterial {
        VertexMaterial { material }
    }

    fn serialize_sparse(
        values: &RangeMap<usize, VertexMaterial>,
        length: usize,
        writer: &mut impl Write,
    ) -> Result<(), SerializeError> {
        let mut last_end = 0;
        for (range, material) in values.iter() {
            serialize_rle(range.start - last_end, &None::<u8>, writer)?;
            serialize_rle(range.clone().count(), &Some(material.material), writer)?;
            last_end = range.end;
        }
        serialize_rle(length - last_end, &None::<u8>, writer)?;
        Ok(())
    }

    fn deserialize_sparse(
        length: usize,
        reader: &mut impl Read,
    ) -> Result<RangeMap<usize, VertexMaterial>, DeserializeError> {
        let mut output = RangeMap::new();
        let mut i = 0;
        while i < length {
            let material = Option::<u8>::deserialize(reader)?;
            let more = u8::deserialize(reader)? as usize + 1;
            if let Some(material) = material {
                output.insert(i..i + more, VertexMaterial { material });
            }
            i += more;
        }
        if i != length {
            return Err(DeserializeError::BadData);
        }
        Ok(output)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, serde::Serialize)]
pub struct VertexVoxel {
    flags: u8,
    position: [u8; 3],
}

impl VertexVoxel {
    pub fn new(position: [u8; 3]) -> VertexVoxel {
        VertexVoxel { flags: 0, position }
    }
}

fn range_intersection(a: &Range<usize>, b: &Range<usize>) -> Range<usize> {
    let start = a.start.max(b.start);
    let end = a.end.min(b.end);
    if start > end {
        end..end
    } else {
        start..end
    }
}

impl VertexVoxel {
    fn serialize_sparse(
        sparse_values: &RangeMap<usize, VertexVoxel>,
        length: usize,
        writer: &mut impl Write,
    ) -> Result<(), SerializeError> {
        let mut flag_ranges = RangeMap::new();
        flag_ranges.insert(0..length, 0);

        for (range, value) in sparse_values.iter() {
            flag_ranges.insert(range.clone(), value.flags | 1)
        }

        for (mut flag_range, flags) in flag_ranges {
            if flags == 0 {
                serialize_rle(flag_range.count(), &flags, writer)?;
            } else {
                while flag_range.clone().count() > 0 {
                    flags.serialize(writer)?;

                    let mut sub_range = flag_range.clone();
                    sub_range.end = sub_range.end.min(sub_range.start + 256);
                    let more = sub_range.clone().count() - 1;
                    (more as u8).serialize(writer)?;

                    for (position_range, position) in sparse_values.overlapping(sub_range.clone()) {
                        serialize_rle(
                            range_intersection(&sub_range, position_range).count(),
                            &position.position,
                            writer,
                        )?;
                    }

                    flag_range.start += 256;
                }
            }
        }
        Ok(())
    }

    fn deserialize_sparse(
        length: usize,
        reader: &mut impl Read,
    ) -> Result<RangeMap<usize, VertexVoxel>, DeserializeError> {
        let mut sparse_vertices = RangeMap::new();
        let mut i = 0;
        while i < length {
            let flags = u8::deserialize(reader)?;
            let masked_flags = flags & 0xFE;
            let more = u8::deserialize(reader)? as usize + 1;
            if flags & 1 != 0 {
                let mut j = 0;
                while j < more {
                    let mut position = [0u8; 3];
                    reader.read_exact(&mut position)?;
                    let yet_more = u8::deserialize(reader)? as usize + 1;
                    let index = i + j;
                    sparse_vertices.insert(
                        index..index + yet_more,
                        VertexVoxel {
                            flags: masked_flags,
                            position,
                        },
                    );
                    j += yet_more;
                }
                if j != more {
                    return Err(DeserializeError::BadData);
                }
            }
            i += more;
        }
        if i != length {
            return Err(DeserializeError::BadData);
        }
        Ok(sparse_vertices)
    }
}

#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct RangeZYX {
    pub origin: Point<i32>,
    pub size: Vector<i32>,
}

impl RangeZYX {
    // The order here matches the order in Aabb::split_at_center()
    pub const OFFSETS: [[i32; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ];

    pub fn with_extent(origin: Point<i32>, extent: i32) -> RangeZYX {
        assert!(extent >= 0);
        RangeZYX::with_extents(origin, Vector::repeat(extent))
    }

    pub fn with_extents(origin: Point<i32>, extents: Vector<i32>) -> RangeZYX {
        assert!(extents.min() >= 0);
        RangeZYX {
            origin,
            size: extents,
        }
    }

    pub fn single(origin: Point<i32>) -> RangeZYX {
        RangeZYX::with_extent(origin, 1)
    }

    pub fn volume(&self) -> u64 {
        self.size.x.abs() as u64 * self.size.y.abs() as u64 * self.size.z.abs() as u64
    }

    pub fn intersection(&self, other: &RangeZYX) -> RangeZYX {
        let origin = self.origin.sup(&other.origin);
        let end = (self.origin + self.size).inf(&(other.origin + other.size));
        let size = (end - origin).sup(&Vector::zeros());
        RangeZYX { origin, size }
    }

    fn for_each_index_range<F>(&self, subrange: &RangeZYX, mut func: F)
    where
        F: FnMut(Range<usize>),
    {
        let intersection = self.intersection(subrange);
        if intersection.size.min() == 0 {
            return;
        }
        for x in 0..intersection.size.x {
            for y in 0..intersection.size.y {
                let start = self.index_from_position(intersection.origin + Vector::new(x, y, 0));
                func(start..start + intersection.size.z as usize)
            }
        }
    }

    fn index_from_relative_position(&self, pos: Vector<i32>) -> usize {
        ((pos.x * self.size.y + pos.y) * self.size.z + pos.z) as usize
    }

    fn index_from_position(&self, pos: Point<i32>) -> usize {
        let offset = pos - self.origin;
        self.index_from_relative_position(offset)
    }

    fn relative_position_from_index(&self, index: usize) -> Vector<i32> {
        let index = index as i32;
        let x_size = self.size.y * self.size.z;
        let x = index / x_size;
        let y = (index % x_size) / self.size.z;
        let z = index % self.size.z;
        Vector::new(x, y, z)
    }

    fn position_from_index(&self, index: usize) -> Point<i32> {
        self.origin + self.relative_position_from_index(index)
    }

    pub fn split_at_center(&self) -> [RangeZYX; 8] {
        let half_extents = self.size / 2;
        Self::OFFSETS.map(|[x, y, z]| {
            let offset = half_extents.component_mul(&Vector::new(x, y, z));
            RangeZYX::with_extents(self.origin + offset, half_extents)
        })
    }
}

impl Serialize for RangeZYX {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        self.origin.serialize(writer)?;
        self.size.serialize(writer)
    }
}

impl Deserialize for RangeZYX {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let origin = Point::deserialize(reader)?;
        let size = Vector::deserialize(reader)?;
        Ok(RangeZYX { origin, size })
    }
}

#[derive(Default, serde::Serialize)]
pub struct VertexGrid {
    range: RangeZYX,
    inner_range: RangeZYX,

    // NQ stores these as dense arrays, but for our purposes this is just easier.
    sparse_materials: RangeMap<usize, VertexMaterial>,
    sparse_vertices: RangeMap<usize, VertexVoxel>,
}

impl VertexGrid {
    const MAGIC: u32 = 0xe881339e;
    const VERSION: u32 = 9;

    pub fn new(range: RangeZYX, inner_range: RangeZYX) -> VertexGrid {
        VertexGrid {
            range,
            inner_range,
            ..Default::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.sparse_materials.is_empty()
    }

    pub fn set_materials(&mut self, subrange: &RangeZYX, material: VertexMaterial) {
        self.range
            .for_each_index_range(subrange, |r| self.sparse_materials.insert(r, material))
    }

    pub fn set_voxel(&mut self, point: &Point<i32>, voxel: VertexVoxel) {
        self.set_voxels(&RangeZYX::single(*point), voxel)
    }

    pub fn set_voxels(&mut self, subrange: &RangeZYX, voxel: VertexVoxel) {
        self.range
            .for_each_index_range(subrange, |r| self.sparse_vertices.insert(r, voxel))
    }

    pub fn calculate_metadata(&self, material_id: u64) -> HeavyMetadata {
        let mut min_pos = Point::new(i32::MAX, i32::MAX, i32::MAX);
        let mut max_pos = Point::new(i32::MIN, i32::MIN, i32::MIN);
        let mut total_materials = 0;
        self.range.for_each_index_range(&self.inner_range, |r| {
            for (subrange, _) in self.sparse_materials.overlapping(&r) {
                for i in range_intersection(subrange, &r) {
                    let pos = self.range.position_from_index(i);
                    min_pos = min_pos.inf(&pos);
                    max_pos = max_pos.sup(&pos);
                    total_materials += 1;
                }
            }
        });
        if total_materials == 0 {
            return HeavyMetadata::default();
        }
        let bounding_box = RangeZYX {
            origin: min_pos,
            size: max_pos - min_pos,
        };
        let mat_count = FixedPoint::from_f64(total_materials as f64 / 64.0);
        let mut material_stats = BTreeMap::new();
        material_stats.insert(
            MaterialId {
                id: material_id,
                short_name: "Material".into(),
            },
            mat_count,
        );
        // We don't actually need to calculate inertia, since the game will do that.
        HeavyMetadata {
            bounding_box: Some(bounding_box),
            material_stats: Some(material_stats),
            inertia: None,
            server_timestamp: 0,
            server_previous_version: 0,
        }
    }
}

impl Debug for VertexGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let materials = Vec::from_iter(
            self.sparse_materials
                .iter()
                .map(|(k, _)| {
                    k.clone()
                        .map(|i| self.range.position_from_index(i).to_string())
                })
                .flatten(),
        );
        let vertices = Vec::from_iter(
            self.sparse_vertices
                .iter()
                .map(|(k, _)| {
                    k.clone()
                        .map(|i| self.range.position_from_index(i).to_string())
                })
                .flatten(),
        );
        f.debug_struct("VertexGrid")
            .field("range", &self.range)
            .field("inner_range", &self.inner_range)
            .field("sparse_materials", &materials)
            .field("sparse_vertices", &vertices)
            .finish()
    }
}

impl Serialize for VertexGrid {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        VertexGrid::MAGIC.serialize(writer)?;
        VertexGrid::VERSION.serialize(writer)?;
        self.range.serialize(writer)?;
        self.inner_range.serialize(writer)?;
        let length = self.range.volume() as usize;
        VertexMaterial::serialize_sparse(&self.sparse_materials, length, writer)?;
        VertexVoxel::serialize_sparse(&self.sparse_vertices, length, writer)?;
        Ok(())
    }
}

impl Deserialize for VertexGrid {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let magic = u32::deserialize(reader)?;
        assert_magic(magic, VertexGrid::MAGIC)?;
        let grid_version = u32::deserialize(reader)?;
        assert_version(grid_version, VertexGrid::VERSION)?;
        let range = RangeZYX::deserialize(reader)?;
        let inner_range = RangeZYX::deserialize(reader)?;
        let length = range.volume() as usize;
        let sparse_materials = VertexMaterial::deserialize_sparse(length, reader)?;
        let sparse_vertices = VertexVoxel::deserialize_sparse(length, reader)?;

        Ok(VertexGrid {
            range,
            inner_range,
            sparse_materials,
            sparse_vertices,
        })
    }
}

#[derive(
    Debug,
    Default,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct MaterialId {
    pub id: u64,
    pub short_name: String,
}

impl Serialize for MaterialId {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        self.id.serialize(writer)?;
        if self.short_name.len() != 8 {
            return Err(SerializeError::BadData);
        }
        writer.write_all(self.short_name.as_bytes())?;
        Ok(())
    }
}

impl Deserialize for MaterialId {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let id = u64::deserialize(reader)?;
        let mut short_name_data = [0; 8];
        reader.read_exact(&mut short_name_data)?;
        Ok(MaterialId {
            id,
            short_name: str::from_utf8(&short_name_data)?.to_string(),
        })
    }
}

#[derive(Debug, Default)]
pub struct MaterialMapper {
    mapping: BTreeMap<MaterialId, u8>,
    reverse_mapping: BTreeMap<u8, MaterialId>,
}

impl MaterialMapper {
    pub fn insert(&mut self, id: u8, material: MaterialId) {
        self.mapping.insert(material.clone(), id);
        self.reverse_mapping.insert(id, material);
    }
}

impl Serialize for MaterialMapper {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        self.mapping.serialize(writer)
    }
}

impl Deserialize for MaterialMapper {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mapping = BTreeMap::<MaterialId, u8>::deserialize(reader)?;
        let reverse_mapping = BTreeMap::from_iter(mapping.iter().map(|(x, y)| (*y, x.clone())));
        Ok(MaterialMapper {
            mapping,
            reverse_mapping,
        })
    }
}

impl serde::Serialize for MaterialMapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("MaterialMapper", 1)?;
        state.serialize_field("mapping", &Vec::from_iter(self.mapping.iter()))?;
        state.end()
    }
}

#[derive(Debug, serde::Serialize)]
pub struct VoxelCellData {
    grid: VertexGrid,
    mapping: MaterialMapper,
    is_diff: u8,
}

impl VoxelCellData {
    const MAGIC: u32 = 0x27b8a013;
    const VERSION: u32 = 6;

    pub fn new(grid: VertexGrid, mapping: MaterialMapper) -> VoxelCellData {
        VoxelCellData {
            grid,
            mapping,
            is_diff: 1,
        }
    }

    pub fn calculate_metadata(&self, hash: i64, material: u64) -> AggregateMetadata {
        let mut light_current = LightMetadata::default();
        let heavy_current = self.grid.calculate_metadata(material);
        light_current.vox = Some(heavy_current.material_stats.is_some());
        light_current.r#mod = Some(true);
        light_current.hash_voxel = Some(hash);
        light_current.hash_decor = Some(0);
        light_current.entropy = Some(0.01); // TODO, calc this
        AggregateMetadata::new(light_current, heavy_current)
    }
}

impl Serialize for VoxelCellData {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        VoxelCellData::MAGIC.serialize(writer)?;
        VoxelCellData::VERSION.serialize(writer)?;
        self.grid.serialize(writer)?;
        self.mapping.serialize(writer)?;
        self.is_diff.serialize(writer)
    }
}

impl Deserialize for VoxelCellData {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let magic = u32::deserialize(reader)?;
        assert_magic(magic, VoxelCellData::MAGIC)?;
        let version = u32::deserialize(reader)?;
        assert_version(version, VoxelCellData::VERSION)?;
        let grid = VertexGrid::deserialize(reader)?;
        let mapping = MaterialMapper::deserialize(reader)?;
        let is_diff = u8::deserialize(reader)?;
        Ok(VoxelCellData {
            grid,
            mapping,
            is_diff,
        })
    }
}

fn maybe_bool_to_int(value: Option<bool>) -> u8 {
    match value {
        Some(true) => 1,
        Some(false) => 2,
        None => 0,
    }
}

fn int_to_maybe_bool(value: u8) -> Option<bool> {
    match value {
        0 => None,
        1 => Some(true),
        2 => Some(false),
        _ => panic!(),
    }
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct LightMetadata {
    vox: Option<bool>,
    r#mod: Option<bool>,
    hash_voxel: Option<i64>,
    hash_decor: Option<i64>,
    entropy: Option<f64>,
}

impl LightMetadata {
    fn combine(&self, other: &LightMetadata) -> LightMetadata {
        LightMetadata {
            vox: match (self.vox, other.vox) {
                (None, None) => None,
                (None, b) => b,
                (a, None) => a,
                (Some(a), Some(b)) => Some(a || b),
            },
            r#mod: Some(false),
            hash_voxel: None,
            hash_decor: None,
            entropy: None,
        }
    }
}

impl Serialize for LightMetadata {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        let vox = maybe_bool_to_int(self.vox);
        let r#mod = maybe_bool_to_int(self.r#mod);
        let value = (vox << 2) | r#mod;
        value.serialize(writer)?;
        self.hash_voxel.serialize(writer)?;
        self.hash_decor.serialize(writer)?;
        self.entropy.serialize(writer)
    }
}

impl Deserialize for LightMetadata {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let value = u8::deserialize(reader)?;
        let vox = int_to_maybe_bool((value >> 2) & 3);
        let r#mod = int_to_maybe_bool(value & 3);

        let hash_voxel = Option::deserialize(reader)?;
        let hash_decor = Option::deserialize(reader)?;
        let entropy = Option::deserialize(reader)?;

        Ok(LightMetadata {
            vox,
            r#mod,
            hash_voxel,
            hash_decor,
            entropy,
        })
    }
}

#[derive(Debug, Default, Clone, serde::Serialize)]
struct Inertia {
    mass: f64,
    gravity_center: Point<f64>,
    inertia_tensor: Vec<f64>, // 6, 3x3 triangular matrix
}

impl Serialize for Inertia {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        self.mass.serialize(writer)?;
        self.gravity_center.serialize(writer)?;
        if self.inertia_tensor.len() != 6 {
            return Err(SerializeError::BadData);
        }
        for v in &self.inertia_tensor {
            v.serialize(writer)?;
        }
        Ok(())
    }
}

impl Deserialize for Inertia {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let mass = f64::deserialize(reader)?;
        let gravity_center = Point::deserialize(reader)?;
        let mut inertia_tensor = Vec::new();
        for _ in 0..6 {
            let value = f64::deserialize(reader)?;
            inertia_tensor.push(value);
        }
        Ok(Inertia {
            mass,
            gravity_center,
            inertia_tensor,
        })
    }
}

#[derive(Clone, Copy, serde::Serialize)]
struct FixedPoint(u64);

impl FixedPoint {
    fn from_f64(value: f64) -> FixedPoint {
        FixedPoint((value / f64::from_bits(0x3e70000000000000)) as u64)
    }

    fn to_f64(&self) -> f64 {
        self.0 as f64 * f64::from_bits(0x3e70000000000000)
    }
}

impl Serialize for FixedPoint {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        self.0.serialize(writer)
    }
}

impl Deserialize for FixedPoint {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        Ok(FixedPoint(u64::deserialize(reader)?))
    }
}

impl Debug for FixedPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FixedPoint").field(&self.to_f64()).finish()
    }
}

#[derive(Debug, Default, Clone)]
pub struct HeavyMetadata {
    pub bounding_box: Option<RangeZYX>,
    material_stats: Option<BTreeMap<MaterialId, FixedPoint>>,
    inertia: Option<Inertia>,
    server_timestamp: u64,
    server_previous_version: u64,
}

impl HeavyMetadata {
    pub fn combine(&self, other: &HeavyMetadata) -> HeavyMetadata {
        let bounding_box = match (&self.bounding_box, &other.bounding_box) {
            (None, None) => None,
            (None, Some(b)) => Some(*b),
            (Some(a), None) => Some(*a),
            (Some(a), Some(b)) => {
                let origin = a.origin.inf(&b.origin);
                let end = (a.origin + a.size).sup(&(b.origin + b.size));
                Some(RangeZYX {
                    origin,
                    size: end - origin,
                })
            }
        };
        let material_stats = match (&self.material_stats, &other.material_stats) {
            (None, None) => None,
            (None, Some(b)) => Some(b.clone()),
            (Some(a), None) => Some(a.clone()),
            (Some(a), Some(b)) => {
                let mut result = a.clone();
                for (k, v) in b {
                    let fp = match result.get(k) {
                        Some(fp) => FixedPoint(v.0 + fp.0),
                        None => *v,
                    };
                    result.insert(k.clone(), fp);
                }
                Some(result)
            }
        };
        HeavyMetadata {
            bounding_box,
            material_stats,
            inertia: None,
            server_timestamp: 0,
            server_previous_version: 0,
        }
    }
}

impl Serialize for HeavyMetadata {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        self.bounding_box.serialize(writer)?;
        self.material_stats.serialize(writer)?;
        self.inertia.serialize(writer)?;
        self.server_timestamp.serialize(writer)?;
        self.server_previous_version.serialize(writer)
    }
}

impl Deserialize for HeavyMetadata {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let bounding_box = Option::deserialize(reader)?;
        let material_stats = Option::deserialize(reader)?;
        let inertia = Option::deserialize(reader)?;
        let server_timestamp = u64::deserialize(reader)? as u64;
        let server_previous_version = u64::deserialize(reader)? as u64;
        Ok(HeavyMetadata {
            bounding_box,
            material_stats,
            inertia,
            server_timestamp,
            server_previous_version,
        })
    }
}

impl serde::Serialize for HeavyMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("HeavyMetadata", 5)?;
        state.serialize_field("bounding_box", &self.bounding_box)?;
        let linear_material_stats = self
            .material_stats
            .as_ref()
            .map(|v| Vec::from_iter(v.iter()));
        state.serialize_field("material_stats", &linear_material_stats)?;
        state.serialize_field("inertia", &self.inertia)?;
        state.serialize_field("server_timestamp", &self.server_timestamp)?;
        state.serialize_field("server_previous_version", &self.server_previous_version)?;
        state.end()
    }
}

#[derive(Debug, serde::Serialize)]
pub struct AggregateMetadata {
    pub light_current: LightMetadata,
    light_children: Vec<LightMetadata>, // 8
    pub heavy_current: HeavyMetadata,
    heavy_children: Vec<HeavyMetadata>, // 8
}

impl AggregateMetadata {
    const MAGIC: u32 = 0x9f81f3c0;
    const VERSION: u32 = 8;

    pub fn new(light_current: LightMetadata, heavy_current: HeavyMetadata) -> AggregateMetadata {
        AggregateMetadata {
            light_current,
            heavy_current,
            ..Default::default()
        }
    }

    pub fn combine(hash: i64, children: &[AggregateMetadata]) -> AggregateMetadata {
        assert_eq!(children.len(), 8);
        let light_children = Vec::from_iter(children.iter().map(|a| a.light_current.clone()));
        let mut light_current = light_children
            .iter()
            .fold(LightMetadata::default(), |acc, n| acc.combine(n));
        light_current.hash_voxel = Some(hash);
        let heavy_children = Vec::from_iter(children.iter().map(|a| a.heavy_current.clone()));
        AggregateMetadata {
            light_current,
            light_children,
            heavy_current: heavy_children
                .iter()
                .fold(HeavyMetadata::default(), |acc, n| acc.combine(n)),
            heavy_children,
        }
    }
}

impl Default for AggregateMetadata {
    fn default() -> Self {
        Self {
            light_current: Default::default(),
            light_children: vec![Default::default(); 8],
            heavy_current: Default::default(),
            heavy_children: vec![Default::default(); 8],
        }
    }
}

impl Serialize for AggregateMetadata {
    fn serialize(&self, writer: &mut impl Write) -> Result<(), SerializeError> {
        AggregateMetadata::MAGIC.serialize(writer)?;
        AggregateMetadata::VERSION.serialize(writer)?;
        self.light_current.serialize(writer)?;
        if self.light_children.len() != 8 {
            return Err(SerializeError::BadData);
        }
        for light in self.light_children.iter() {
            light.serialize(writer)?;
        }
        self.heavy_current.serialize(writer)?;
        if self.heavy_children.len() != 8 {
            return Err(SerializeError::BadData);
        }
        for heavy in self.heavy_children.iter() {
            heavy.serialize(writer)?;
        }
        Ok(())
    }
}

impl Deserialize for AggregateMetadata {
    fn deserialize(reader: &mut impl Read) -> Result<Self, DeserializeError> {
        let magic = u32::deserialize(reader)?;
        assert_magic(magic, AggregateMetadata::MAGIC)?;
        let version = u32::deserialize(reader)?;
        assert_version(version, AggregateMetadata::VERSION)?;
        let light_current = LightMetadata::deserialize(reader)?;
        let mut light_children = Vec::with_capacity(8);
        for _ in 0..8 {
            let light_child = LightMetadata::deserialize(reader)?;
            light_children.push(light_child);
        }
        let heavy_current = HeavyMetadata::deserialize(reader)?;
        let mut heavy_children = Vec::with_capacity(8);
        for _ in 0..8 {
            let heavy_child = HeavyMetadata::deserialize(reader)?;
            heavy_children.push(heavy_child);
        }
        Ok(AggregateMetadata {
            light_current,
            light_children,
            heavy_current,
            heavy_children,
        })
    }
}

pub fn hash(bytes: &[u8]) -> i64 {
    xxhash_rust::xxh64::xxh64(&bytes, 0xa1b2c3d4e5f6e7d8) as i64
}
