use std::{iter, ops::Index};

use itertools::Itertools;
use ndarray::{array, azip, Array2};

use crate::game::{neighbors, CellState, Field};

use self::bitvec_bitgrid::BitGrid;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Prediction {
    Contradiction,
    Free,
    Mine,
    Probability(f32, u8),
}

impl Prediction {
    fn combine(self, other: Self) -> Self {
        match (self, other) {
            // Free and Mine is a Contradiction, and anything with a Contradiction is a Contradiction
            (Self::Contradiction, _)
            | (_, Self::Contradiction)
            | (Self::Free, Self::Mine)
            | (Self::Mine, Self::Free) => Self::Contradiction,
            // Both Free or both Mine become that
            result @ ((Self::Free, Self::Free) | (Self::Mine, Self::Mine)) => result.0,
            // Free or Mine win out over a Probability
            (result @ (Self::Free | Self::Mine), Self::Probability(..))
            | (Self::Probability(..), result @ (Self::Free | Self::Mine)) => result,
            // Average two Probabilities into a new Probability
            (Self::Probability(p1, n1), Self::Probability(p2, n2)) => {
                let n = n1 + n2;
                Self::Probability((p1 * (n1 as f32) + p2 * (n2 as f32)) / n as f32, n)
            }
        }
    }

    pub(crate) fn from_probability(prob: f32) -> Self {
        match prob {
            p if p == 0.0 => Self::Free,
            p if p == 1.0 => Self::Mine,
            p => Self::Probability(p, 0),
        }
    }
}

pub(crate) fn predict(field: &Field) -> Array2<Option<f32>> {
    let mut regions = iter::once(Region::from_field_unrevealed(field))
        .chain(
            field
                .board
                .indexed_iter()
                .filter_map(|(pos, _)| Region::from_cell_revealed(field, pos)),
        )
        .collect::<Vec<_>>();

    let mut predictions = Array2::<Option<Option<f32>>>::default(field.size());

    'outer: loop {
        // regions.retain(|region| {
        //     let probability = if region.is_clear() {
        //         0.0
        //     } else if region.is_full() {
        //         1.0
        //     } else {
        //         return true;
        //     };
        //     println!("{region:#?}");
        //     azip!((pred in &mut predictions, &c in &region.region) if c { *pred = Some(Some(probability)) });
        //     false
        // });

        for ((a_i, a), (b_i, b)) in regions.iter().enumerate().tuple_combinations() {
            if let Some(new_regions) = a.split_overlap(b) {
                regions.remove(b_i);
                regions.remove(a_i);
                regions.extend(new_regions.into_iter().filter(|region| region.size != 0));

                continue 'outer;
            }
        }

        break;
    }

    for region in regions {
        let probability = region.mines as f32 / region.size as f32;
        for pos in region.region.indices() {
            match predictions[pos] {
                Some(Some(prev_prob)) if prev_prob != probability => predictions[pos] = Some(None),
                None => predictions[pos] = Some(Some(probability)),
                _ => {}
            }
        }
    }

    predictions.mapv_into_any(Option::flatten)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Region {
    region: BitGrid,
    size: usize,
    mines: usize,
}

impl Region {
    fn empty(size: (usize, usize)) -> Self {
        Self {
            region: BitGrid::empty(size),
            size: 0,
            mines: 0,
        }
    }

    fn from_field_unrevealed(field: &Field) -> Self {
        let region = BitGrid::empty(field.size()).with_indices(
            field
                .board
                .indexed_iter()
                .filter_map(|(pos, cell)| (cell.state == CellState::Unrevealed).then_some(pos)),
        );
        let size = region.size();
        Self {
            region,
            size,
            mines: field.remaining_mines(),
        }
    }

    fn from_cell_revealed(field: &Field, pos: (usize, usize)) -> Option<Self> {
        if field.board[pos].state != CellState::Revealed {
            return None;
        }

        let mut region = Self::empty(field.size());
        region.mines = field.board[pos].neighbors as usize;
        for neighbor_pos in neighbors(&field.board, pos) {
            match field.board[neighbor_pos].state {
                CellState::Flagged => region.mines -= 1,
                CellState::Unrevealed => {
                    region.size += 1;
                    region.region.set(neighbor_pos, true);
                }
                _ => {}
            }
        }

        Some(region)
    }

    fn is_clear(&self) -> bool {
        self.mines == 0
    }

    fn is_full(&self) -> bool {
        self.size == self.mines
    }

    fn merge_full(&self, other: &Self) -> Option<Self> {
        if !self.is_full() || !other.is_full() {
            return None;
        }

        let size = self.size + other.size - (&self.region & &other.region).size();

        Some(Self {
            region: &self.region | &other.region,
            size,
            mines: size,
        })
    }

    fn split_overlap(&self, other: &Self) -> Option<[Self; 3]> {
        // Assumed prerequisite: each region does not have more mines than they have space to
        // actually contain
        // 1. If the regions don't overlap, return None
        // 2. Split into 3 parts: a without b, b without a, and the overlap
        // 3. Find the possible ways to split the mines from a into a and overlap, and b into b and overlap
        // 4 If there is only one possible way that both sides agree on, then do that
        let a = self;
        let b = other;

        let a_only = &a.region & &!&b.region;
        let b_only = &b.region & &!&a.region;
        let overlap = &a.region & &b.region;
        let overlap_size = overlap.size();
        if overlap_size == 0 {
            return None;
        }

        let a_range = a.mines.saturating_sub(a.size - overlap_size)..=a.mines.min(overlap_size);
        let b_range = b.mines.saturating_sub(b.size - overlap_size)..=b.mines.min(overlap_size);
        let overlap_mines = if a_range.start() == b_range.end() {
            *a_range.start()
        } else if a_range.end() == b_range.start() {
            *a_range.end()
        } else {
            return None;
        };

        Some([
            Self {
                region: a_only,
                size: a.size - overlap_size,
                mines: a.mines - overlap_mines,
            },
            Self {
                region: overlap,
                size: overlap_size,
                mines: overlap_mines,
            },
            Self {
                region: b_only,
                size: b.size - overlap_size,
                mines: b.mines - overlap_mines,
            },
        ])
    }
}

mod ndarray_bitgrid {
    use std::ops::{BitAnd, BitOr, Index, IndexMut, Not};

    use ndarray::Array2;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BitGrid(Array2<bool>);

    impl BitGrid {
        pub fn empty(size: (usize, usize)) -> Self {
            Self(Array2::default(size))
        }

        pub fn size(&self) -> usize {
            self.0.iter().filter(|c| **c).count()
        }

        pub fn set(&mut self, pos: (usize, usize), value: bool) {
            self.0[pos] = value;
        }

        pub fn indices(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
            self.0
                .indexed_iter()
                .filter_map(|(pos, c)| c.then_some(pos))
        }

        pub fn with_indices(mut self, indices: impl Iterator<Item = (usize, usize)>) -> Self {
            for pos in indices {
                self.set(pos, true);
            }
            self
        }
    }

    impl Index<(usize, usize)> for BitGrid {
        type Output = bool;

        fn index(&self, index: (usize, usize)) -> &Self::Output {
            &self.0[index]
        }
    }

    impl BitAnd<&BitGrid> for &BitGrid {
        type Output = BitGrid;

        fn bitand(self, rhs: &BitGrid) -> Self::Output {
            BitGrid(&self.0 & &rhs.0)
        }
    }

    impl BitOr<&BitGrid> for &BitGrid {
        type Output = BitGrid;

        fn bitor(self, rhs: &BitGrid) -> Self::Output {
            BitGrid(&self.0 | &rhs.0)
        }
    }

    impl Not for &BitGrid {
        type Output = BitGrid;

        fn not(self) -> Self::Output {
            BitGrid(!&self.0)
        }
    }
}

mod bitvec_bitgrid {
    use std::ops::{BitAnd, BitOr, Index, IndexMut, Not};

    use bitvec::vec::BitVec;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BitGrid {
        grid: BitVec,
        stride: usize,
    }

    impl BitGrid {
        pub fn empty(size: (usize, usize)) -> Self {
            BitGrid {
                grid: bitvec::bitvec![0; size.0 * size.1],
                stride: size.1,
            }
        }

        pub fn size(&self) -> usize {
            self.grid.count_ones()
        }

        pub fn set(&mut self, pos: (usize, usize), value: bool) {
            self.grid.set(pos.0 * self.stride + pos.1, value);
        }

        pub fn indices(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
            self.grid
                .iter_ones()
                .map(|i| (i / self.stride, i % self.stride))
        }

        pub fn with_indices(mut self, indices: impl Iterator<Item = (usize, usize)>) -> Self {
            for pos in indices {
                self.set(pos, true);
            }
            self
        }
    }

    impl Index<(usize, usize)> for BitGrid {
        type Output = bool;

        fn index(&self, index: (usize, usize)) -> &Self::Output {
            &self.grid[index.0 * self.stride + index.1]
        }
    }

    impl BitAnd<&BitGrid> for &BitGrid {
        type Output = BitGrid;

        fn bitand(self, rhs: &BitGrid) -> Self::Output {
            BitGrid {
                grid: self.grid.clone() & &rhs.grid,
                stride: self.stride,
            }
        }
    }

    impl BitOr<&BitGrid> for &BitGrid {
        type Output = BitGrid;

        fn bitor(self, rhs: &BitGrid) -> Self::Output {
            BitGrid {
                grid: self.grid.clone() | &rhs.grid,
                stride: self.stride,
            }
        }
    }

    impl Not for &BitGrid {
        type Output = BitGrid;

        fn not(self) -> Self::Output {
            BitGrid {
                grid: self.grid.clone().not(),
                stride: self.stride,
            }
        }
    }
}

// #[test]
// fn test_regions() {
//     let test1 = Region {
//         region: array![[true, true, true, false]],
//         size: 3,
//         mines: 2,
//     }
//     .split_overlap(&Region {
//         region: array![[false, true, true, true]],
//         size: 3,
//         mines: 1,
//     });
//     dbg!(test1);
//
//     let test2 = Region {
//         region: array![[true, true, false]],
//         size: 2,
//         mines: 2,
//     }
//     .split_overlap(&Region {
//         region: array![[true, true, true]],
//         size: 3,
//         mines: 2,
//     });
//     dbg!(test2);
// }
