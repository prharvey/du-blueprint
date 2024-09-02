use std::fmt::Debug;

use parry3d_f64::math::Point;

use crate::squarion::*;

#[derive(Debug, Clone)]
enum SvoNode<T> {
    Leaf(Option<T>),
    Internal(Box<[SvoNode<T>; 8]>),
}

impl<T> PartialEq for SvoNode<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(l0), Self::Leaf(r0)) => l0 == r0,
            _ => false,
        }
    }
}

pub enum SvoReturn<T> {
    Leaf(Option<T>),
    Continue,
}

impl<T> SvoNode<T>
where
    T: Clone + PartialEq,
{
    fn from_fn<F>(range: &RangeZYX, func: &F) -> Self
    where
        F: Fn(&RangeZYX) -> SvoReturn<T>,
    {
        assert!(range.size.min() != 0);
        match func(range) {
            SvoReturn::Leaf(v) => SvoNode::Leaf(v),
            SvoReturn::Continue => SvoNode::Internal(Box::new(
                range.split_at_center().map(|o| Self::from_fn(&o, func)),
            )),
        }
    }

    fn fold_volume<Acc, F>(&self, range: &RangeZYX, volume: &RangeZYX, acc: Acc, func: &F) -> Acc
    where
        F: Fn(Acc, &RangeZYX, &T) -> Acc,
    {
        let intersection = volume.intersection(range);
        match intersection.volume() {
            0 => acc,
            _ => match self {
                SvoNode::Leaf(v) => match v {
                    Some(v) => func(acc, &intersection, v),
                    None => acc,
                },
                SvoNode::Internal(children) => children
                    .iter()
                    .zip(range.split_at_center())
                    .fold(acc, |acc, (node, subrange)| {
                        node.fold_volume(&subrange, volume, acc, func)
                    }),
            },
        }
    }
}

pub struct Svo<T> {
    root: SvoNode<T>,
    pub range: RangeZYX,
}

impl<T> Svo<T>
where
    T: Clone + PartialEq,
{
    pub fn from_fn<F>(origin: Point<i32>, extent: usize, func: &F) -> Self
    where
        F: Fn(&RangeZYX) -> SvoReturn<T>,
    {
        assert!(extent.is_power_of_two());
        let range = RangeZYX::with_extent(origin, extent as i32);
        Self {
            root: SvoNode::from_fn(&range, func),
            range,
        }
    }

    pub fn fold<Acc, F>(&self, acc: Acc, func: &F) -> Acc
    where
        F: Fn(Acc, &RangeZYX, &T) -> Acc,
    {
        self.fold_volume(&self.range, acc, func)
    }

    pub fn fold_volume<Acc, F>(&self, volume: &RangeZYX, acc: Acc, func: &F) -> Acc
    where
        F: Fn(Acc, &RangeZYX, &T) -> Acc,
    {
        self.root.fold_volume(&self.range, volume, acc, func)
    }
}
