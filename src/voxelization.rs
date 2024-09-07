use async_std::task::{self, block_on};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use line_drawing::{VoxelOrigin, WalkVoxels};
use ordered_float::NotNan;
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Isometry, Point, Vector};
use parry3d_f64::query::{intersection_test, PointQuery};
use parry3d_f64::shape::{Cuboid, Shape, TriMesh, Triangle};

use crate::squarion::*;
use crate::svo::*;

#[derive(Debug, Clone, PartialEq)]
enum Voxel {
    Internal,
    External,
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
            return SvoReturn::Leaf(Voxel::External);
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
                SvoReturn::Leaf(Voxel::Internal)
            } else {
                SvoReturn::Leaf(Voxel::External)
            }
        } else if range.volume() == 1 {
            // We do a quick check to see if the voxel is "significant", i.e. the center is in the mesh.
            //
            // This helps remove artifacts from internal angles in the model.
            let significant = mesh.contains_point(isometry, &aabb.center());
            SvoReturn::Leaf(Voxel::Boundry(significant))
        } else {
            SvoReturn::Internal(Voxel::Boundry(false))
        }
    })
}

fn discretize(point: Point<f64>, voxel_size: f64) -> Point<f64> {
    (84.0 * point / voxel_size).map(|v| v.round())
}

// In game voxels operate on a discrete grid, so the best solutions are ones that
// minimize error when going from the model surface to the in game voxel surface.
fn lowest_error_point_on_surface(
    starts: &[Point<f64>],
    end: &Point<f64>,
    shape: &impl Shape,
) -> Point<f64> {
    let mut lowest_error = f64::MAX;
    let mut best = *end;

    for start in starts {
        if (end - start).magnitude() < 0.1 {
            continue;
        }
        let dir = (end - start).normalize();
        let start = end - 5.0 * dir;
        let end = end + 5.0 * dir;
        for (x, y, z) in WalkVoxels::<f64, i64>::new(
            (start.x, start.y, start.z),
            (end.x, end.y, end.z),
            &VoxelOrigin::Corner,
        ) {
            let point = Point::new(x as f64, y as f64, z as f64);
            let error = shape.distance_to_local_point(&point, false);
            if error < lowest_error {
                lowest_error = error;
                best = point;
            }
        }
    }
    best
}

fn to_voxel_offset(offset: Vector<f64>) -> Vector<u8> {
    offset.map(|v| (126.0 + v).clamp(0.0, 252.0) as u8)
}

// Tries to snap to the nearest vertex, then edge, then face.
//
// Results in some artifacts in game where too many voxels snap to a vertex/edge and as a result
// you get some weird shadows on flat surfaces, but this otherwise preserves the model features.
fn calculate_vertex_offset(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    anchor: Point<f64>,
    voxel_size: f64,
) -> Vector<u8> {
    let discrete_anchor = discretize(anchor, voxel_size);
    let discrete_pos = discretize(aabb.center(), voxel_size);
    let (_, feature) = mesh.project_point_and_get_feature(isometry, &anchor);
    let face = feature.unwrap_face();
    let original_triangle = mesh.triangle(face).transformed(isometry);
    let a = discretize(original_triangle.a, voxel_size);
    let b = discretize(original_triangle.b, voxel_size);
    let c = discretize(original_triangle.c, voxel_size);
    let triangle = Triangle::new(a, b, c);

    let closest_vertex = triangle
        .vertices()
        .iter()
        .min_by_key(|v| NotNan::new((*v - discrete_anchor).magnitude()).unwrap())
        .unwrap();
    let offset = closest_vertex - discrete_pos;
    if offset.magnitude() < 84.0 {
        to_voxel_offset(closest_vertex - discrete_pos)
    } else {
        let (segment, closest_edge) = triangle
            .edges()
            .map(|s| (s, s.project_local_point(&discrete_anchor, false).point))
            .into_iter()
            .min_by_key(|(_, v)| NotNan::new((*v - discrete_anchor).magnitude()).unwrap())
            .unwrap();
        let offset = closest_edge - discrete_pos;
        if offset.magnitude() < 84.0 {
            let best = lowest_error_point_on_surface(&[segment.a], &closest_edge, &segment);
            to_voxel_offset(best - discrete_pos)
        } else {
            // Detect a degenerate triangle.
            if triangle.area() <= 1e-6 {
                let fallback = original_triangle
                    .project_local_point(&discrete_anchor, false)
                    .point;
                to_voxel_offset(fallback - discrete_pos)
            } else {
                let point = triangle.project_local_point(&discrete_anchor, false).point;
                let best = lowest_error_point_on_surface(&[a, b, c], &point, &triangle);
                to_voxel_offset(best - discrete_pos)
            }
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
    let mut significant_points = HashMap::new();
    voxels.cata(|range, v, cs| {
        if cs.is_some() {
            return;
        }
        match v {
            Voxel::Boundry(significant) => {
                assert_eq!(range.volume(), 1);
                let center =
                    aabb.mins + voxel_size * (range.origin - origin).map(|v| v as f64 + 0.5);
                for offset in &RangeZYX::OFFSETS {
                    let offset = Vector::from_row_slice(offset);
                    let point = range.origin + offset;

                    let pos = aabb.mins + voxel_size * (point - origin).map(|v| v as f64);
                    if mesh.contains_point(isometry, &pos) != *significant {
                        let entry = significant_points
                            .entry(point)
                            .or_insert_with(|| Vec::new());
                        entry.push(center.coords)
                    }
                }
            }
            _ => (),
        };
    });

    let mut result = HashMap::new();
    for (point, anchors) in significant_points {
        let anchor = anchors.iter().fold(Point::origin(), |a, v| a + v) / anchors.len() as f64;
        let pos = aabb.mins + voxel_size * (point - origin).map(|v| v as f64);
        let aabb = Aabb::from_half_extents(pos, Vector::repeat(voxel_size * 1.5));

        let best = calculate_vertex_offset(isometry, mesh, &aabb, anchor, voxel_size);
        result.insert(point, Point::origin() + best);
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
    is_lod: bool,
) -> Option<VoxelCellData> {
    // We have to over-voxelize that chunk due to the boundries expected in voxel cell data.
    // e.g. for an inner_range of [0, 0, 0] -> [32, 32, 32] the actual range of the chunk is
    //  [-1, -1, -1] -> [34, 34, 34], likely to remove seams when generating the mesh.
    let voxel_size = aabb.extents().x / 32.0;
    let voxel_size_offset = Vector::repeat(voxel_size);
    let origin = aabb.mins - voxel_size_offset * 2.0;

    let range = RangeZYX::with_extent(voxel_origin - Vector::repeat(1), 35);

    // Note that this large aabb could result in a lot of wasted computation, so we clip the range.
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
    let mut grid = VertexGrid::new(range, inner_range);
    voxels.cata(|subrange, value, cs| {
        if cs.is_some() {
            return;
        }
        let (place_materials, place_positions) = match value {
            Voxel::External => (false, false),
            Voxel::Internal => (true, true),
            Voxel::Boundry(significant) => (*significant, true),
        };
        if place_materials {
            // Materials are placed on the +[1, 1, 1] vertex.
            let material_range = RangeZYX {
                origin: subrange.origin + Vector::repeat(1),
                size: subrange.size,
            };
            grid.set_materials(&material_range, VertexMaterial::new(2));
        }
        if place_positions {
            // Set the default positions for all voxels. We will update the significant ones later.
            let voxel_range = RangeZYX {
                origin: subrange.origin,
                size: subrange.size + Vector::repeat(1),
            };
            grid.set_voxels(&voxel_range, VertexVoxel::new([126, 126, 126]));
        }
    });

    if !is_lod && grid.is_empty() {
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

pub struct Voxelizer {
    isometry: Arc<Isometry<f64>>,
    mesh: Arc<TriMesh>,
}

impl Voxelizer {
    pub fn new(isometry: Isometry<f64>, mesh: TriMesh) -> Voxelizer {
        Voxelizer {
            isometry: Arc::new(isometry),
            mesh: Arc::new(mesh),
        }
    }

    pub fn create_lods(
        &self,
        aabb: &Aabb,
        origin: Point<i32>,
        height: usize,
        material: u64,
    ) -> Svo<Option<VoxelCellData>> {
        let extent = 1 << height;
        let chunk_size = aabb.extents().x / extent as f64;
        let chunk_futures = Svo::from_fn(origin, extent, &|range| {
            let mins = aabb.mins + (range.origin - origin).map(|v| v as f64) * chunk_size;
            let maxs = mins + range.size.map(|v| v as f64) * chunk_size;
            let aabb = Aabb::new(mins.into(), maxs.into());

            let cuboid = Cuboid::new(aabb.half_extents() * 1.05);
            let cuboid_pos = Isometry::from(aabb.center());
            if intersection_test(&self.isometry, self.mesh.as_ref(), &cuboid_pos, &cuboid).unwrap()
            {
                let is_lod = range.size.x > 1;
                let voxel_origin = range.origin * 32 / range.size.x;
                let isometry = self.isometry.clone();
                let mesh = self.mesh.clone();
                let task = task::spawn(async move {
                    voxelize_chunk(
                        &isometry,
                        &mesh,
                        &aabb,
                        &voxel_origin,
                        material,
                        is_lod,
                    )
                });
                if range.size.x == 1 {
                    SvoReturn::Leaf(Some(task))
                } else {
                    SvoReturn::Internal(Some(task))
                }
            } else {
                SvoReturn::Leaf(None)
            }
        });

        chunk_futures.into_map(|f| f.map(|f| block_on(f)).flatten())
    }
}
