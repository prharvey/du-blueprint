use std::{array, fmt::Debug};

use parry3d_f64::math::Point;

use crate::squarion::*;

#[derive(Debug)]
enum SvoNode<T> {
    Leaf(T),
    Internal(T, Box<[SvoNode<T>; 8]>),
}

pub enum SvoReturn<T> {
    Leaf(T),
    Internal(T),
}

impl<T> SvoNode<T> {
    fn from_fn<F>(range: &RangeZYX, func: &F) -> Self
    where
        F: Fn(&RangeZYX) -> SvoReturn<T>,
    {
        assert!(range.size.min() != 0);
        match func(range) {
            SvoReturn::Leaf(v) => SvoNode::Leaf(v),
            SvoReturn::Internal(v) => SvoNode::Internal(
                v,
                Box::new(range.split_at_center().map(|o| Self::from_fn(&o, func))),
            ),
        }
    }

    fn cata<F, R>(&self, range: &RangeZYX, func: &mut F) -> R
    where
        F: FnMut(&RangeZYX, &T, Option<[R; 8]>) -> R,
    {
        match self {
            SvoNode::Leaf(v) => func(range, v, None),
            SvoNode::Internal(v, children) => {
                let octants = range.split_at_center();
                let results = array::from_fn(|i| children[i].cata(&octants[i], func));
                func(range, v, Some(results))
            }
        }
    }

    fn into_cata<F, R>(self, range: &RangeZYX, func: &mut F) -> R
    where
        F: FnMut(&RangeZYX, T, Option<[R; 8]>) -> R,
    {
        match self {
            SvoNode::Leaf(v) => func(range, v, None),
            SvoNode::Internal(v, children) => {
                let octants = range.split_at_center();
                // This is the only good way to move out of an array. It's kinda dumb.
                let mut i = 0;
                let results = children.map(|c| {
                    let result = c.into_cata(&octants[i], func);
                    i += 1;
                    result
                });
                func(range, v, Some(results))
            }
        }
    }
}

pub struct Svo<T> {
    root: SvoNode<T>,
    pub range: RangeZYX,
}

impl<T> Svo<T> {
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

    pub fn cata<F, R>(&self, mut func: F) -> R
    where
        F: FnMut(&RangeZYX, &T, Option<[R; 8]>) -> R,
    {
        self.root.cata(&self.range, &mut func)
    }

    pub fn into_map<F, R>(self, mut func: F) -> Svo<R>
    where
        F: FnMut(T) -> R,
    {
        Svo {
            root: self.root.into_cata(&self.range, &mut |_, v, cs| match cs {
                Some(cs) => SvoNode::Internal(func(v), Box::new(cs)),
                None => SvoNode::Leaf(func(v)),
            }),
            range: self.range,
        }
    }
}
