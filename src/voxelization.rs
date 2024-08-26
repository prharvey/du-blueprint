use core::f64;
use std::collections::HashSet;
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

#[derive(Debug, Clone)]
enum Voxel {
    Inside,
    Outside,
    Boundry(bool),
}

// The order here matches the order in Aabb::split_at_center()
const OFFSETS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

#[derive(Debug, Clone)]
enum Svo {
    Leaf(Voxel),
    Internal(Box<[Svo; 8]>),
}

impl Svo {
    fn voxelize(isometry: &Isometry<f64>, mesh: &TriMesh, aabb: &Aabb, height: usize) -> Self {
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
                return Svo::Leaf(Voxel::Inside);
            } else {
                return Svo::Leaf(Voxel::Outside);
            }
        }
        if height == 0 {
            // We do a quick check to see if the voxel is "significant", i.e. the center is in the mesh
            // or the middle of the voxel intersects the mesh.
            //
            // This helps remove artifacts from internal angles in the model.
            let sphere = Ball::new(aabb.half_extents().x / 1.3);
            let sphere_pos = Isometry::from(aabb.center());
            let significant = mesh.contains_point(isometry, &aabb.center())
                || intersection_test(isometry, mesh, &sphere_pos, &sphere).unwrap();
            return Svo::Leaf(Voxel::Boundry(significant));
        }
        let octants = aabb
            .split_at_center()
            .map(|octant| Svo::voxelize(isometry, mesh, &octant, height - 1));
        Svo::Internal(Box::new(octants))
    }

    fn extract_vertices(
        &self,
        isometry: &Isometry<f64>,
        mesh: &TriMesh,
        aabb: &Aabb,
        origin: Point<i32>,
        extent: i32,
        processed: &mut HashSet<Point<i32>>,
        grid: &mut VertexGrid,
    ) {
        match self {
            Svo::Leaf(v) => match v {
                Voxel::Inside => {
                    grid.set_materials(
                        &RangeZYX::with_extent(origin + Vector::repeat(1), extent),
                        VertexMaterial::new(2),
                    );
                    // Unfortunately we need to set the position of the voxels here due to non-manifold meshes.
                    // Otherwise the result will not be editable in game.
                    //
                    // This should just be the surface area, but this is easier and it doesn't really matter.
                    for x in 0..extent + 1 {
                        for y in 0..extent + 1 {
                            for z in 0..extent + 1 {
                                let point = &origin + Vector::new(x, y, z);
                                if !processed.contains(&point) {
                                    grid.set_voxel(&point, VertexVoxel::new([126, 126, 126]));
                                }
                            }
                        }
                    }
                }
                Voxel::Outside => (),
                Voxel::Boundry(significant) => {
                    assert_eq!(extent, 1);
                    if *significant {
                        grid.set_material(&(origin + Vector::repeat(1)), VertexMaterial::new(2));
                    }
                    let voxel_size = aabb.extents().x;
                    for offset in &OFFSETS {
                        let offset = Vector::from_row_slice(offset);
                        let point = &origin + offset;
                        if processed.contains(&point) {
                            continue;
                        }
                        let pos = aabb.mins + voxel_size * offset.map(|v| v as f64);
                        if mesh.contains_point(isometry, &pos) != *significant {
                            processed.insert(point);

                            let aabb =
                                Aabb::from_half_extents(pos, Vector::repeat(voxel_size * 1.5));

                            let best = calculate_vertex_position(isometry, mesh, &aabb);
                            let offset = 84.0 * (best - pos) / voxel_size;
                            let coord =
                                offset.map(|v| (126 + v.round() as i32).clamp(0, 252) as u8);
                            grid.set_voxel(&point, VertexVoxel::new([coord.x, coord.y, coord.z]));
                        } else {
                            grid.set_voxel(&point, VertexVoxel::new([126, 126, 126]));
                        }
                    }
                }
            },
            Svo::Internal(children) => {
                let octants = aabb.split_at_center();
                for i in 0..8 {
                    let offset = Vector::from_row_slice(&OFFSETS[i]);
                    let half_extent = extent / 2;
                    let origin = origin + offset * half_extent;
                    Svo::extract_vertices(
                        &children[i],
                        isometry,
                        mesh,
                        &octants[i],
                        origin,
                        half_extent,
                        processed,
                        grid,
                    )
                }
            }
        }
    }
}

// This is by far the most expensive part, mostly due to Trimesh being kinda slow and the algorithm itself
// being pretty naive. For now we just throw threads at it, but it can definitely be improved.
fn voxelize_chunk(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    voxel_origin: &Point<i32>,
    material : u64
) -> Option<VoxelCellData> {
    // We have to over-voxelize that chunk due to weird boundries expected in voxel cell data.
    // e.g. for an inner_range of [0, 0, 0] -> [32, 32, 32] the actual range of the chunk is
    //  [-1, -1, -1] -> [34, 34, 34], likely to remove seems when generating the mesh.
    let voxel_size = aabb.extents().x / 32.0;
    let voxel_size_offset = Vector::repeat(voxel_size);
    let origin = aabb.mins - voxel_size_offset * 2.0;

    // Note that this large aabb results in a lot of wasted computation.
    let svo_aabb = Aabb::new(origin, origin + voxel_size_offset * 64.0);
    let svo = Svo::voxelize(isometry, mesh, &svo_aabb, 6);

    let range = RangeZYX::with_extent(voxel_origin - Vector::repeat(1), 35);
    let inner_range = RangeZYX::with_extent(*voxel_origin, 32);
    let mut grid = VertexGrid::new(range, inner_range);
    let mut processed = HashSet::new();
    svo.extract_vertices(
        isometry,
        mesh,
        &svo_aabb,
        range.origin - Vector::repeat(1),
        64,
        &mut processed,
        &mut grid,
    );

    if grid.is_empty() {
        return None;
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
        material : u64
    ) -> LodNode {
        if height == 0 {
            return LodNode::Leaf(voxelize_chunk(isometry, mesh, aabb, &(origin * 32), material));
        }
        let extent = 1 << height;
        let parent_origin = origin / extent as i32;
        let octants = aabb.split_at_center();
        let (me, children) = rayon::join(
            || voxelize_chunk(isometry, mesh, aabb, &(parent_origin * 32), material),
            || {
                octants
                    .par_iter()
                    .zip(&OFFSETS)
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
        material : u64,
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
                let children =
                    Vec::from_iter(children.iter().zip(&OFFSETS).map(|(child, offset)| {
                        let half_extent = extent / 2;
                        let origin = coords + Vector::from_row_slice(offset) * half_extent;
                        child.make_voxel_data(origin, height - 1, material, result)
                    }));
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
    pub fn voxelize(isometry: &Isometry<f64>, mesh: &TriMesh, aabb: &Aabb, height: usize, material : u64) -> Lod {
        Lod {
            root: LodNode::voxelize(isometry, mesh, &aabb, Point::origin(), height, material),
            height,
        }
    }

    pub fn make_voxel_data(&self, material : u64) -> (Vec<VoxelData>, RangeZYX) {
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
}

impl Blueprint {
    pub fn new(
        name: String,
        size: usize,
        core_id: u64,
        bounds: Aabb,
        voxel_data: Vec<VoxelData>,
    ) -> Blueprint {
        Blueprint {
            name,
            size,
            core_id,
            bounds,
            voxel_data,
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
                "Static": false,
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
