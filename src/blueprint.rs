use base64::prelude::*;
use chrono::prelude::*;
use parry3d_f64::math::{Point, Vector};
use serde_json::json;

use crate::squarion::*;
use crate::svo::*;

struct VoxelData {
    pub height: usize,
    pub coords: Point<i32>,
    pub meta_data: String,
    pub meta_hash: i64,
    pub voxel_data: String,
    pub voxel_hash: i64,
}

impl VoxelData {
    fn to_construct_json(&self) -> serde_json::Value {
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

fn make_voxel_data(data: &Svo<Option<VoxelCellData>>, material: u64) -> (Vec<VoxelData>, RangeZYX) {
    let mut result = Vec::new();
    let meta = data.cata(|range, v, cs| match v {
        Some(voxels) => {
            let voxel_compressed = voxels.compress().unwrap();
            let voxel_hash = hash(&voxel_compressed);
            let voxel_data = BASE64_STANDARD.encode(voxel_compressed);

            let meta = match cs {
                Some(cs) => AggregateMetadata::combine(voxel_hash, &cs),
                None => voxels.calculate_metadata(voxel_hash, material),
            };
            let meta_compressed = meta.compress().unwrap();
            let meta_hash = hash(&meta_compressed);
            let meta_data = BASE64_STANDARD.encode(meta_compressed);

            let extent = range.size.x;
            let coords = range.origin / extent;
            let height = (extent as u32).trailing_zeros() as usize;
            result.push(VoxelData {
                height,
                coords,
                meta_data,
                meta_hash,
                voxel_data,
                voxel_hash,
            });
            meta
        }
        None => AggregateMetadata::default(),
    });
    (result, meta.heavy_current.bounding_box.unwrap())
}

pub struct Blueprint {
    name: String,
    size: usize,
    core_id: u64,
    fill_material: u64,
    voxel_data: Svo<Option<VoxelCellData>>,
    is_static: bool,
}

impl Blueprint {
    pub fn new(
        name: String,
        size: usize,
        core_id: u64,
        fill_material: u64,
        voxel_data: Svo<Option<VoxelCellData>>,
        is_static: bool,
    ) -> Blueprint {
        Blueprint {
            name,
            size,
            core_id,
            fill_material,
            voxel_data,
            is_static,
        }
    }

    pub fn to_construct_json(&self) -> serde_json::Value {
        let (voxel_data, bb) = make_voxel_data(&self.voxel_data, self.fill_material);
        let mins = bb.origin.map(|v| v as f64) / 4.0;
        let maxs = mins + bb.size.map(|v| v as f64) / 4.0;
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
                    "min": { "x": mins.x, "y": mins.y, "z": mins.z },
                    "max": { "x": maxs.x, "y": maxs.y, "z": maxs.z }
                },
                "FreeDeploy": false,
                "MaxUse": null,
                "HasMaterials": false,
                "DataId": null
            },
            "VoxelData": serde_json::Value::Array(voxel_data.iter().map(VoxelData::to_construct_json).collect()),
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
