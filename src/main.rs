use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use base64::Engine;
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Isometry, Point, Vector};
use parry3d_f64::shape::{TriMesh, TriMeshFlags};
use squarion::{AggregateMetadata, Deserialize, VoxelCellData};
use tobj::LoadOptions;

mod blueprint;
mod squarion;
mod svo;
mod voxelization;

use crate::blueprint::*;
use crate::voxelization::*;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct ScaleInfo {
    /// Automatically scale model to fill core
    #[arg(short, long)]
    auto: bool,

    #[arg(long, default_value_t = 1.0)]
    scale: f64,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a blueprint file from an obj file.
    Generate {
        /// Input obj file name
        input: PathBuf,

        /// Output blueprint file name
        output: PathBuf,

        #[arg(short, long, value_enum)]
        r#type: CoreType,

        #[arg(short, long, value_enum)]
        size: CoreSize,

        /// Voxel material ID
        #[arg(short, long, default_value_t = 1971262921)]
        material: u64,

        #[command(flatten)]
        scale: ScaleInfo,
    },
    /// Parse a base64 voxel chunk and dump the result to stdout
    ParseVoxel {
        // Input base64
        b64: String,
    },
    /// Parse a base64 meta chunk and dump the result to stdout
    ParseMeta {
        // Input base64
        b64: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate {
            input,
            output,
            size,
            r#type,
            material,
            scale,
        } => {
            let (models, _) = tobj::load_obj(
                &input,
                &LoadOptions {
                    merge_identical_points: true,
                    triangulate: true,
                    ..Default::default()
                },
            )
            .unwrap();

            let mut mesh: Option<TriMesh> = None;
            for model in models {
                let vertices = Vec::from_iter(
                    model
                        .mesh
                        .positions
                        .chunks_exact(3)
                        .map(|x| Point::from_slice(&[x[0] as f64, x[1] as f64, x[2] as f64])),
                );
                let indices = Vec::from_iter(
                    model
                        .mesh
                        .indices
                        .chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]]),
                );
                let sub_mesh = TriMesh::new(vertices, indices);
                match &mut mesh {
                    Some(mesh) => mesh.append(&sub_mesh),
                    None => mesh = Some(sub_mesh),
                }
            }
            let mut mesh = mesh.unwrap();
            mesh.set_flags(
                TriMeshFlags::ORIENTED
                    | TriMeshFlags::FIX_INTERNAL_EDGES
                    | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES,
            )
            .unwrap();

            // TODO: allow translations and rotations
            let isometry = Isometry::default();

            let height = size.height() - 3;
            let aabb = mesh.aabb(&isometry);
            let svo_aabb = if scale.auto {
                let scale = Vector::repeat(aabb.extents().max()).component_div(&aabb.extents());
                aabb.scaled_wrt_center(&scale)
                    .scaled_wrt_center(&Vector::repeat(2.0))
            } else {
                let extents = Vector::repeat(4.0 * (1 << height) as f64);
                Aabb::from_half_extents(aabb.center(), extents / scale.scale)
            };

            let voxelizer = Voxelizer::new(isometry, mesh);
            let svo = voxelizer.create_lods(&svo_aabb, Point::origin(), height, material);
            let bp = Blueprint::new(
                input
                    .clone()
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
                CoreInfo::from(size, r#type),
                material,
                svo,
            );
            File::create(output)
                .unwrap()
                .write(bp.to_construct_json().to_string().as_bytes())
                .unwrap();
        }
        Commands::ParseVoxel { b64 } => {
            let bytes = base64::prelude::BASE64_STANDARD.decode(b64).unwrap();
            let voxel = VoxelCellData::decompress(&bytes);
            match voxel {
                Ok(voxel) => println!("{:#?}", serde_json::to_string(&voxel)),
                Err(_) => println!("{:#?}", voxel),
            }
        }
        Commands::ParseMeta { b64 } => {
            let bytes = base64::prelude::BASE64_STANDARD.decode(b64).unwrap();
            let meta = AggregateMetadata::decompress(&bytes);
            match meta {
                Ok(meta) => println!("{:#?}", serde_json::to_string(&meta)),
                Err(_) => println!("{:#?}", meta),
            }
        }
    }
}
