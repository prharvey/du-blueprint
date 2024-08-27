use core::f64;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use base64::prelude::*;
use chrono::prelude::*;
use ordered_float::NotNan;
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Isometry, Point, Vector};
use parry3d_f64::query::{intersection_test, PointQuery};
use parry3d_f64::shape::{Ball, Cuboid, TriMesh};
use rayon::prelude::*;
use serde_json::json;

use crate::squarion::*;
use crate::svo::*;

#[derive(Debug, Clone, PartialEq)]
enum Voxel {
    Internal,
    Boundry(bool),
}

fn voxelize(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    extent: usize,
    clip_range: &RangeZYX,
) -> Svo<Voxel> {
    let voxel_size = aabb.extents().x / extent as f64;
    Svo::from_fn(origin, extent, &|range| {
        if range.intersection(clip_range).volume() == 0 {
            return SvoReturn::Leaf(None);
        }

        let mins = aabb.mins + (range.origin - origin).map(|v| v as f64) * voxel_size;
        let maxs = mins + range.size.map(|v| v as f64) * voxel_size;
        let aabb = Aabb::new(mins, maxs);

        // Scale up the region slightly. Makes intersection detection more robust.
        let cuboid = Cuboid::new(aabb.half_extents() * 1.05);
        let cuboid_pos = Isometry::from(aabb.center());
        if !intersection_test(isometry, mesh, &cuboid_pos, &cuboid).unwrap() {
            // Vote on if the voxel is inside or outside. We need to do this because some people won't
            // read the FAQ, and try to import non-manifold meshes. This makes the process more reliable.
            let mut inside_count = mesh.contains_point(isometry, &aabb.center()) as u32;
            for point in aabb.vertices() {
                inside_count += mesh.contains_point(isometry, &point) as u32
            }
            // Bias towards assuming outside, since it's better to have empty internals than random
            // floating cubes.
            if inside_count >= 7 {
                return SvoReturn::Leaf(Some(Voxel::Internal));
            } else {
                return SvoReturn::Leaf(None);
            }
        }
        if range.volume() == 1 {
            // We do a quick check to see if the voxel is "significant", i.e. the center is in the mesh
            // or the middle of the voxel intersects the mesh.
            //
            // This helps remove artifacts from internal angles in the model.
            let sphere = Ball::new(aabb.half_extents().x / 1.3);
            let sphere_pos = Isometry::from(aabb.center());
            let significant = mesh.contains_point(isometry, &aabb.center())
                || intersection_test(isometry, mesh, &sphere_pos, &sphere).unwrap();
            return SvoReturn::Leaf(Some(Voxel::Boundry(significant)));
        }
        SvoReturn::Continue
    })
}

// Tries to snap to the nearest vertex, then edge, then face.
//
// Results in some artifacts in game where too many voxels snap to a vertex/edge and as a result
// you get some weird shadows on flat surfaces, but this otherwise preserves the model features.
fn calculate_vertex_position(isometry: &Isometry<f64>, mesh: &TriMesh, aabb: &Aabb) -> Point<f64> {
    let pos = aabb.center();

    let (projection, feature) = mesh.project_point_and_get_feature(isometry, &pos);
    let face = feature.unwrap_face();
    let triangle = mesh.triangle(face);

    let closest_vertex = triangle
        .vertices()
        .iter()
        .map(|p| isometry * p)
        .min_by_key(|v| NotNan::new((*v - pos).magnitude()).unwrap())
        .unwrap();
    if aabb.contains_local_point(&closest_vertex) {
        closest_vertex
    } else {
        let closest_edge = triangle
            .edges()
            .iter()
            .map(|s| s.project_point(isometry, &pos, false).point)
            .min_by_key(|v| NotNan::new((*v - pos).magnitude()).unwrap())
            .unwrap();
        if aabb.contains_local_point(&closest_edge) {
            closest_edge
        } else {
            projection.point
        }
    }
}

fn extract_vertices(
    voxels: &Svo<Voxel>,
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
) -> HashMap<Point<i32>, Point<u8>> {
    let voxel_size = aabb.extents().x / voxels.range.size.x as f64;
    let significant_points = voxels.fold(HashSet::new(), &|mut acc, range, v| {
        match v {
            Voxel::Internal => (),
            Voxel::Boundry(significant) => {
                assert_eq!(range.volume(), 1);
                for offset in &RangeZYX::OFFSETS {
                    let offset = Vector::from_row_slice(offset);
                    let point = &range.origin + offset;

                    let pos = aabb.mins + voxel_size * (point - origin).map(|v| v as f64);
                    if mesh.contains_point(isometry, &pos) != *significant {
                        acc.insert(point);
                    }
                }
            }
        };
        acc
    });

    let mut result = HashMap::new();
    for point in significant_points {
        let pos = aabb.mins + voxel_size * (point - origin).map(|v| v as f64);
        let aabb = Aabb::from_half_extents(pos, Vector::repeat(voxel_size * 1.5));

        let best = calculate_vertex_position(isometry, mesh, &aabb);
        let offset = 84.0 * (best - pos) / voxel_size;
        let coord = offset.map(|v| (126 + v.round() as i32).clamp(0, 252) as u8);
        result.insert(point, Point::origin() + coord);
    }
    result
}

// This is by far the most expensive part, mostly due to Trimesh being kinda slow and the algorithm itself
// being pretty naive. For now we just throw threads at it, but it can definitely be improved.
fn voxelize_chunk(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    voxel_origin: &Point<i32>,
    material: u64,
) -> Option<VoxelCellData> {
    // We have to over-voxelize that chunk due to the boundries expected in voxel cell data.
    // e.g. for an inner_range of [0, 0, 0] -> [32, 32, 32] the actual range of the chunk is
    //  [-1, -1, -1] -> [34, 34, 34], likely to remove seams when generating the mesh.
    let voxel_size = aabb.extents().x / 32.0;
    let voxel_size_offset = Vector::repeat(voxel_size);
    let origin = aabb.mins - voxel_size_offset * 2.0;

    let range = RangeZYX::with_extent(voxel_origin - Vector::repeat(1), 35);

    // Note that this large aabb could result in a lot of wasted computation, so we pass in the range above
    // to restrict the expensive computation.
    let svo_aabb = Aabb::new(origin, origin + voxel_size_offset * 64.0);
    let voxels = voxelize(
        isometry,
        mesh,
        &svo_aabb,
        voxel_origin - Vector::repeat(2),
        64,
        &range,
    );

    let inner_range = RangeZYX::with_extent(*voxel_origin, 32);
    let mut grid = voxels.fold(
        VertexGrid::new(range, inner_range),
        &|mut grid, subrange, value| {
            let place_materials = match value {
                Voxel::Internal => true,
                Voxel::Boundry(significant) => *significant,
            };
            if place_materials {
                // Materials are placed on the +[1, 1, 1] vertex.
                let material_range = RangeZYX {
                    origin: subrange.origin + Vector::repeat(1),
                    size: subrange.size,
                };
                grid.set_materials(&material_range, VertexMaterial::new(2));
            }

            // Set the default positions for all voxels. We will update the significant ones later.
            let voxel_range = RangeZYX {
                origin: subrange.origin,
                size: subrange.size + Vector::repeat(1),
            };
            grid.set_voxels(&voxel_range, VertexVoxel::new([126, 126, 126]));
            grid
        },
    );

    if grid.is_empty() {
        return None;
    }

    // Extract the non-default vertices and set them now.
    let vertices = extract_vertices(
        &voxels,
        isometry,
        mesh,
        &svo_aabb,
        voxel_origin - Vector::repeat(2),
    );
    for (point, offset) in vertices {
        grid.set_voxel(&point, VertexVoxel::new([offset.x, offset.y, offset.z]));
    }

    let mut mapping = MaterialMapper::default();

    // Every blueprint I checked had this debug material in the first index.
    // I assume there is a reason for it, so we'll add it as well.
    mapping.insert(
        1,
        MaterialId {
            id: 157903047,
            short_name: "Debug1\0\0".into(),
        },
    );
    mapping.insert(
        2,
        MaterialId {
            id: material,
            short_name: "Material".into(),
        },
    );

    Some(VoxelCellData::new(grid, mapping))
}

#[derive(Debug)]
enum LodNode {
    Leaf(Option<VoxelCellData>),
    Internal(Option<VoxelCellData>, Box<[LodNode; 8]>),
}

impl LodNode {
    fn voxelize(
        isometry: &Isometry<f64>,
        mesh: &TriMesh,
        aabb: &Aabb,
        origin: Point<i32>,
        height: usize,
        material: u64,
    ) -> LodNode {
        if height == 0 {
            return LodNode::Leaf(voxelize_chunk(
                isometry,
                mesh,
                aabb,
                &(origin * 32),
                material,
            ));
        }
        let extent = 1 << height;
        let parent_origin = origin / extent as i32;
        let octants = aabb.split_at_center();
        let (me, children) = rayon::join(
            || voxelize_chunk(isometry, mesh, aabb, &(parent_origin * 32), material),
            || {
                octants
                    .par_iter()
                    .zip(&RangeZYX::OFFSETS)
                    .map(|(aabb, offset)| {
                        let offset = Vector::from_row_slice(offset);
                        let half_extent = extent / 2;
                        let origin = origin + offset * half_extent as i32;
                        LodNode::voxelize(isometry, mesh, &aabb, origin, height - 1, material)
                    })
                    .collect::<Vec<_>>()
            },
        );
        LodNode::Internal(me, Box::new(children.try_into().unwrap()))
    }

    fn make_voxel_data(
        &self,
        coords: Point<i32>,
        height: usize,
        material: u64,
        result: &mut Vec<VoxelData>,
    ) -> AggregateMetadata {
        match self {
            LodNode::Leaf(Some(voxels)) => {
                let voxels_compressed = voxels.compress().unwrap();
                let voxels_hash = hash(&voxels_compressed);
                let voxels_b64 = BASE64_STANDARD.encode(voxels_compressed);

                let meta = voxels.calculate_metadata(voxels_hash, material);
                let meta_compressed = meta.compress().unwrap();
                let meta_hash = hash(&meta_compressed);
                let meta_b64 = BASE64_STANDARD.encode(meta_compressed);

                let data = VoxelData {
                    height,
                    coords,
                    meta_data: meta_b64,
                    meta_hash,
                    voxel_data: voxels_b64,
                    voxel_hash: voxels_hash,
                };
                result.push(data);
                meta
            }
            LodNode::Internal(Some(voxels), children) => {
                let voxels_compressed = voxels.compress().unwrap();
                let voxels_hash = hash(&voxels_compressed);
                let voxels_b64 = BASE64_STANDARD.encode(voxels_compressed);

                let extent = 1 << height;
                let children = Vec::from_iter(children.iter().zip(&RangeZYX::OFFSETS).map(
                    |(child, offset)| {
                        let half_extent = extent / 2;
                        let origin = coords + Vector::from_row_slice(offset) * half_extent;
                        child.make_voxel_data(origin, height - 1, material, result)
                    },
                ));
                let meta = AggregateMetadata::combine(voxels_hash, &children);
                let meta_compressed = meta.compress().unwrap();
                let meta_hash = hash(&meta_compressed);
                let meta_b64 = BASE64_STANDARD.encode(meta_compressed);

                let data = VoxelData {
                    height: height,
                    coords: coords / extent,
                    meta_data: meta_b64,
                    meta_hash,
                    voxel_data: voxels_b64,
                    voxel_hash: voxels_hash,
                };
                result.push(data);
                meta
            }
            _ => AggregateMetadata::default(),
        }
    }
}

pub struct Lod {
    root: LodNode,
    height: usize,
}

impl Lod {
    pub fn voxelize(
        isometry: &Isometry<f64>,
        mesh: &TriMesh,
        aabb: &Aabb,
        height: usize,
        material: u64,
    ) -> Lod {
        Lod {
            root: LodNode::voxelize(isometry, mesh, &aabb, Point::origin(), height, material),
            height,
        }
    }

    pub fn make_voxel_data(&self, material: u64) -> (Vec<VoxelData>, RangeZYX) {
        let mut result = Vec::new();
        let meta = self
            .root
            .make_voxel_data(Point::origin(), self.height, material, &mut result);
        (result, meta.heavy_current.bounding_box.unwrap())
    }
}

pub struct VoxelData {
    pub height: usize,
    pub coords: Point<i32>,
    pub meta_data: String,
    pub meta_hash: i64,
    pub voxel_data: String,
    pub voxel_hash: i64,
}

impl VoxelData {
    pub fn to_construct_json(&self) -> serde_json::Value {
        json!({
          "created_at": { "$date": Utc::now().to_rfc3339() },
          "h": self.height + 3,
          "k": 0,
          "metadata": { "mversion": 0, "pversion": 4, "uptodate": true, "version": 8 },
          "oid": { "$numberLong": 1 },
          "records": {
            "meta": {
              "data": { "$binary": self.meta_data, "$type": "0x0" },
              "hash": { "$numberLong": self.meta_hash },
              "updated_at": { "$date": Utc::now().to_rfc3339() }
            },
            "voxel": {
              "data": { "$binary": self.voxel_data, "$type": "0x0" },
              "hash": { "$numberLong": self.voxel_hash },
              "updated_at": { "$date": Utc::now().to_rfc3339() }
            }
          },
          "updated_at": { "$date": Utc::now().to_rfc3339() },
          "version": { "$numberLong": 1 },
          "x": { "$numberLong": self.coords.x },
          "y": { "$numberLong": self.coords.y },
          "z": { "$numberLong": self.coords.z }
        })
    }
}

pub struct Blueprint {
    name: String,
    size: usize,
    core_id: u64,
    bounds: Aabb,
    voxel_data: Vec<VoxelData>,
    is_static: bool,
}

impl Blueprint {
    pub fn new(
        name: String,
        size: usize,
        core_id: u64,
        bounds: Aabb,
        voxel_data: Vec<VoxelData>,
        is_static: bool,
    ) -> Blueprint {
        Blueprint {
            name,
            size,
            core_id,
            bounds,
            voxel_data,
            is_static,
        }
    }

    pub fn to_construct_json(&self) -> serde_json::Value {
        let center = Vector::repeat(self.size as f32 / 2.0 + 0.125);
        json!({
            "Model": {
                "Id": 1,
                "Name": self.name,
                "Size": self.size,
                "CreatedAt": Utc::now().to_rfc3339(),
                "CreatorId": 2,
                "JsonProperties": {
                    "kind": 4,
                    "size": self.size,
                    "serverProperties": {
                        "creatorId": { "playerId": 2, "organizationId": 0 },
                        "originConstructId": 1,
                        "blueprintId": null,
                        "isFixture": null,
                        "isBase": null,
                        "isFlaggedForModeration": null,
                        "isDynamicWreck": false,
                        "fuelType": null,
                        "fuelAmount": null,
                        "rdmsTags": { "constructTags": [  ], "elementsTags": [  ] },
                        "compacted": false,
                        "dynamicFixture": null,
                        "constructCloneSource": null
                    },
                    "header": null,
                    "voxelGeometry": { "size": self.size, "kind": 1, "voxelLod0": 3, "radius": null, "minRadius": null, "maxRadius": null },
                    "planetProperties": null,
                    "isNPC": false,
                    "isUntargetable": false
                },
                "Static": self.is_static,
                "Bounds": {
                    "min": { "x": self.bounds.mins.x, "y": self.bounds.mins.y, "z": self.bounds.mins.z },
                    "max": { "x": self.bounds.maxs.x, "y": self.bounds.maxs.y, "z": self.bounds.maxs.z }
                },
                "FreeDeploy": false,
                "MaxUse": null,
                "HasMaterials": false,
                "DataId": null
            },
            "VoxelData": serde_json::Value::Array(self.voxel_data.iter().map(VoxelData::to_construct_json).collect()),
            "Elements": [
              {
                "elementId": 1,
                "localId": 1,
                "constructId": 0,
                "playerId": 0,
                "elementType": self.core_id,
                "position": { "x": center.x, "y": center.y, "z": center.z },
                "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 },
                "properties": [ [ "drmProtected", { "type": 1, "value": false } ] ],
                "serverProperties": {  },
                "links": [  ]
              }
            ],
            "Links": [  ]
        })
    }
}
